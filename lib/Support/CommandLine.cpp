//===-- CommandLine.cpp - Command line parser implementation --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements a command line argument processor that is useful when
// creating a tool.  It provides a simple, minimalistic interface that is easily
// extensible and supports nonlocal (library) command line options.
//
// Note that rather than trying to figure out what this code does, you could try
// reading the library documentation located in docs/CommandLine.html
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Streams.h"
#include "llvm/System/Path.h"
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <cstdlib>
#include <cerrno>
#include <cstring>
using namespace llvm;
using namespace cl;

//===----------------------------------------------------------------------===//
// Template instantiations and anchors.
//
TEMPLATE_INSTANTIATION(class basic_parser<bool>);
TEMPLATE_INSTANTIATION(class basic_parser<int>);
TEMPLATE_INSTANTIATION(class basic_parser<unsigned>);
TEMPLATE_INSTANTIATION(class basic_parser<double>);
TEMPLATE_INSTANTIATION(class basic_parser<float>);
TEMPLATE_INSTANTIATION(class basic_parser<std::string>);

TEMPLATE_INSTANTIATION(class opt<unsigned>);
TEMPLATE_INSTANTIATION(class opt<int>);
TEMPLATE_INSTANTIATION(class opt<std::string>);
TEMPLATE_INSTANTIATION(class opt<bool>);

void Option::anchor() {}
void basic_parser_impl::anchor() {}
void parser<bool>::anchor() {}
void parser<int>::anchor() {}
void parser<unsigned>::anchor() {}
void parser<double>::anchor() {}
void parser<float>::anchor() {}
void parser<std::string>::anchor() {}

//===----------------------------------------------------------------------===//

// Globals for name and overview of program.  Program name is not a string to
// avoid static ctor/dtor issues.
static char ProgramName[80] = "<premain>";
static const char *ProgramOverview = 0;

// This collects additional help to be printed.
static ManagedStatic<std::vector<const char*> > MoreHelp;

extrahelp::extrahelp(const char *Help)
  : morehelp(Help) {
  MoreHelp->push_back(Help);
}

//===----------------------------------------------------------------------===//
// Basic, shared command line option processing machinery.
//

static ManagedStatic<std::map<std::string, Option*> > Options;
static ManagedStatic<std::vector<Option*> > PositionalOptions;

static Option *getOption(const std::string &Str) {
  std::map<std::string,Option*>::iterator I = Options->find(Str);
  return I != Options->end() ? I->second : 0;
}

static void AddArgument(const char *ArgName, Option *Opt) {
  if (getOption(ArgName)) {
    llvm_cerr << ProgramName << ": CommandLine Error: Argument '"
              << ArgName << "' defined more than once!\n";
  } else {
    // Add argument to the argument map!
    (*Options)[ArgName] = Opt;
  }
}

// RemoveArgument - It's possible that the argument is no longer in the map if
// options have already been processed and the map has been deleted!
//
static void RemoveArgument(const char *ArgName, Option *Opt) {
  if (Options->empty()) return;

#ifndef NDEBUG
  // This disgusting HACK is brought to you courtesy of GCC 3.3.2, which ICE's
  // If we pass ArgName directly into getOption here.
  std::string Tmp = ArgName;
  assert(getOption(Tmp) == Opt && "Arg not in map!");
#endif
  Options->erase(ArgName);
}

static inline bool ProvideOption(Option *Handler, const char *ArgName,
                                 const char *Value, int argc, char **argv,
                                 int &i) {
  // Enforce value requirements
  switch (Handler->getValueExpectedFlag()) {
  case ValueRequired:
    if (Value == 0) {       // No value specified?
      if (i+1 < argc) {     // Steal the next argument, like for '-o filename'
        Value = argv[++i];
      } else {
        return Handler->error(" requires a value!");
      }
    }
    break;
  case ValueDisallowed:
    if (Value)
      return Handler->error(" does not allow a value! '" +
                            std::string(Value) + "' specified.");
    break;
  case ValueOptional:
    break;
  default:
    llvm_cerr << ProgramName
              << ": Bad ValueMask flag! CommandLine usage error:"
              << Handler->getValueExpectedFlag() << "\n";
    abort();
    break;
  }

  // Run the handler now!
  return Handler->addOccurrence(i, ArgName, Value ? Value : "");
}

static bool ProvidePositionalOption(Option *Handler, const std::string &Arg,
                                    int i) {
  int Dummy = i;
  return ProvideOption(Handler, Handler->ArgStr, Arg.c_str(), 0, 0, Dummy);
}


// Option predicates...
static inline bool isGrouping(const Option *O) {
  return O->getFormattingFlag() == cl::Grouping;
}
static inline bool isPrefixedOrGrouping(const Option *O) {
  return isGrouping(O) || O->getFormattingFlag() == cl::Prefix;
}

// getOptionPred - Check to see if there are any options that satisfy the
// specified predicate with names that are the prefixes in Name.  This is
// checked by progressively stripping characters off of the name, checking to
// see if there options that satisfy the predicate.  If we find one, return it,
// otherwise return null.
//
static Option *getOptionPred(std::string Name, unsigned &Length,
                             bool (*Pred)(const Option*)) {

  Option *Op = getOption(Name);
  if (Op && Pred(Op)) {
    Length = Name.length();
    return Op;
  }

  if (Name.size() == 1) return 0;
  do {
    Name.erase(Name.end()-1, Name.end());   // Chop off the last character...
    Op = getOption(Name);

    // Loop while we haven't found an option and Name still has at least two
    // characters in it (so that the next iteration will not be the empty
    // string...
  } while ((Op == 0 || !Pred(Op)) && Name.size() > 1);

  if (Op && Pred(Op)) {
    Length = Name.length();
    return Op;             // Found one!
  }
  return 0;                // No option found!
}

static bool RequiresValue(const Option *O) {
  return O->getNumOccurrencesFlag() == cl::Required ||
         O->getNumOccurrencesFlag() == cl::OneOrMore;
}

static bool EatsUnboundedNumberOfValues(const Option *O) {
  return O->getNumOccurrencesFlag() == cl::ZeroOrMore ||
         O->getNumOccurrencesFlag() == cl::OneOrMore;
}

/// ParseCStringVector - Break INPUT up wherever one or more
/// whitespace characters are found, and store the resulting tokens in
/// OUTPUT. The tokens stored in OUTPUT are dynamically allocated
/// using strdup (), so it is the caller's responsibility to free ()
/// them later.
///
static void ParseCStringVector(std::vector<char *> &output,
                               const char *input) {
  // Characters which will be treated as token separators:
  static const char *delims = " \v\f\t\r\n";

  std::string work (input);
  // Skip past any delims at head of input string.
  size_t pos = work.find_first_not_of (delims);
  // If the string consists entirely of delims, then exit early.
  if (pos == std::string::npos) return;
  // Otherwise, jump forward to beginning of first word.
  work = work.substr (pos);
  // Find position of first delimiter.
  pos = work.find_first_of (delims);

  while (!work.empty() && pos != std::string::npos) {
    // Everything from 0 to POS is the next word to copy.
    output.push_back (strdup (work.substr (0,pos).c_str ()));
    // Is there another word in the string?
    size_t nextpos = work.find_first_not_of (delims, pos + 1);
    if (nextpos != std::string::npos) {
      // Yes? Then remove delims from beginning ...
      work = work.substr (work.find_first_not_of (delims, pos + 1));
      // and find the end of the word.
      pos = work.find_first_of (delims);
    } else {
      // No? (Remainder of string is delims.) End the loop.
      work = "";
      pos = std::string::npos;
    }
  }

  // If `input' ended with non-delim char, then we'll get here with
  // the last word of `input' in `work'; copy it now.
  if (!work.empty ()) {
    output.push_back (strdup (work.c_str ()));
  }
}

/// ParseEnvironmentOptions - An alternative entry point to the
/// CommandLine library, which allows you to read the program's name
/// from the caller (as PROGNAME) and its command-line arguments from
/// an environment variable (whose name is given in ENVVAR).
///
void cl::ParseEnvironmentOptions(const char *progName, const char *envVar,
                                 const char *Overview) {
  // Check args.
  assert(progName && "Program name not specified");
  assert(envVar && "Environment variable name missing");

  // Get the environment variable they want us to parse options out of.
  const char *envValue = getenv(envVar);
  if (!envValue)
    return;

  // Get program's "name", which we wouldn't know without the caller
  // telling us.
  std::vector<char*> newArgv;
  newArgv.push_back(strdup(progName));

  // Parse the value of the environment variable into a "command line"
  // and hand it off to ParseCommandLineOptions().
  ParseCStringVector(newArgv, envValue);
  int newArgc = newArgv.size();
  ParseCommandLineOptions(newArgc, &newArgv[0], Overview);

  // Free all the strdup()ed strings.
  for (std::vector<char*>::iterator i = newArgv.begin(), e = newArgv.end();
       i != e; ++i)
    free (*i);
}

/// LookupOption - Lookup the option specified by the specified option on the
/// command line.  If there is a value specified (after an equal sign) return
/// that as well.
static Option *LookupOption(const char *&Arg, const char *&Value) {
  while (*Arg == '-') ++Arg;  // Eat leading dashes

  const char *ArgEnd = Arg;
  while (*ArgEnd && *ArgEnd != '=')
    ++ArgEnd; // Scan till end of argument name.

  if (*ArgEnd == '=')  // If we have an equals sign...
    Value = ArgEnd+1;  // Get the value, not the equals


  if (*Arg == 0) return 0;

  // Look up the option.
  std::map<std::string, Option*> &Opts = *Options;
  std::map<std::string, Option*>::iterator I =
    Opts.find(std::string(Arg, ArgEnd));
  return (I != Opts.end()) ? I->second : 0;
}

void cl::ParseCommandLineOptions(int &argc, char **argv,
                                 const char *Overview) {
  assert((!Options->empty() || !PositionalOptions->empty()) &&
         "No options specified, or ParseCommandLineOptions called more"
         " than once!");
  sys::Path progname(argv[0]);

  // Copy the program name into ProgName, making sure not to overflow it.
  std::string ProgName = sys::Path(argv[0]).getLast();
  if (ProgName.size() > 79) ProgName.resize(79);
  strcpy(ProgramName, ProgName.c_str());
  
  ProgramOverview = Overview;
  bool ErrorParsing = false;

  std::map<std::string, Option*> &Opts = *Options;
  std::vector<Option*> &PositionalOpts = *PositionalOptions;

  // Check out the positional arguments to collect information about them.
  unsigned NumPositionalRequired = 0;
  
  // Determine whether or not there are an unlimited number of positionals
  bool HasUnlimitedPositionals = false;
  
  Option *ConsumeAfterOpt = 0;
  if (!PositionalOpts.empty()) {
    if (PositionalOpts[0]->getNumOccurrencesFlag() == cl::ConsumeAfter) {
      assert(PositionalOpts.size() > 1 &&
             "Cannot specify cl::ConsumeAfter without a positional argument!");
      ConsumeAfterOpt = PositionalOpts[0];
    }

    // Calculate how many positional values are _required_.
    bool UnboundedFound = false;
    for (unsigned i = ConsumeAfterOpt != 0, e = PositionalOpts.size();
         i != e; ++i) {
      Option *Opt = PositionalOpts[i];
      if (RequiresValue(Opt))
        ++NumPositionalRequired;
      else if (ConsumeAfterOpt) {
        // ConsumeAfter cannot be combined with "optional" positional options
        // unless there is only one positional argument...
        if (PositionalOpts.size() > 2)
          ErrorParsing |=
            Opt->error(" error - this positional option will never be matched, "
                       "because it does not Require a value, and a "
                       "cl::ConsumeAfter option is active!");
      } else if (UnboundedFound && !Opt->ArgStr[0]) {
        // This option does not "require" a value...  Make sure this option is
        // not specified after an option that eats all extra arguments, or this
        // one will never get any!
        //
        ErrorParsing |= Opt->error(" error - option can never match, because "
                                   "another positional argument will match an "
                                   "unbounded number of values, and this option"
                                   " does not require a value!");
      }
      UnboundedFound |= EatsUnboundedNumberOfValues(Opt);
    }
    HasUnlimitedPositionals = UnboundedFound || ConsumeAfterOpt;
  }

  // PositionalVals - A vector of "positional" arguments we accumulate into
  // the process at the end...
  //
  std::vector<std::pair<std::string,unsigned> > PositionalVals;

  // If the program has named positional arguments, and the name has been run
  // across, keep track of which positional argument was named.  Otherwise put
  // the positional args into the PositionalVals list...
  Option *ActivePositionalArg = 0;

  // Loop over all of the arguments... processing them.
  bool DashDashFound = false;  // Have we read '--'?
  for (int i = 1; i < argc; ++i) {
    Option *Handler = 0;
    const char *Value = 0;
    const char *ArgName = "";

    // Check to see if this is a positional argument.  This argument is
    // considered to be positional if it doesn't start with '-', if it is "-"
    // itself, or if we have seen "--" already.
    //
    if (argv[i][0] != '-' || argv[i][1] == 0 || DashDashFound) {
      // Positional argument!
      if (ActivePositionalArg) {
        ProvidePositionalOption(ActivePositionalArg, argv[i], i);
        continue;  // We are done!
      } else if (!PositionalOpts.empty()) {
        PositionalVals.push_back(std::make_pair(argv[i],i));

        // All of the positional arguments have been fulfulled, give the rest to
        // the consume after option... if it's specified...
        //
        if (PositionalVals.size() >= NumPositionalRequired &&
            ConsumeAfterOpt != 0) {
          for (++i; i < argc; ++i)
            PositionalVals.push_back(std::make_pair(argv[i],i));
          break;   // Handle outside of the argument processing loop...
        }

        // Delay processing positional arguments until the end...
        continue;
      }
    } else if (argv[i][0] == '-' && argv[i][1] == '-' && argv[i][2] == 0 &&
               !DashDashFound) {
      DashDashFound = true;  // This is the mythical "--"?
      continue;              // Don't try to process it as an argument itself.
    } else if (ActivePositionalArg &&
               (ActivePositionalArg->getMiscFlags() & PositionalEatsArgs)) {
      // If there is a positional argument eating options, check to see if this
      // option is another positional argument.  If so, treat it as an argument,
      // otherwise feed it to the eating positional.
      ArgName = argv[i]+1;
      Handler = LookupOption(ArgName, Value);
      if (!Handler || Handler->getFormattingFlag() != cl::Positional) {
        ProvidePositionalOption(ActivePositionalArg, argv[i], i);
        continue;  // We are done!
      }

    } else {     // We start with a '-', must be an argument...
      ArgName = argv[i]+1;
      Handler = LookupOption(ArgName, Value);

      // Check to see if this "option" is really a prefixed or grouped argument.
      if (Handler == 0) {
        std::string RealName(ArgName);
        if (RealName.size() > 1) {
          unsigned Length = 0;
          Option *PGOpt = getOptionPred(RealName, Length, isPrefixedOrGrouping);

          // If the option is a prefixed option, then the value is simply the
          // rest of the name...  so fall through to later processing, by
          // setting up the argument name flags and value fields.
          //
          if (PGOpt && PGOpt->getFormattingFlag() == cl::Prefix) {
            Value = ArgName+Length;
            assert(Opts.find(std::string(ArgName, Value)) != Opts.end() &&
                   Opts.find(std::string(ArgName, Value))->second == PGOpt);
            Handler = PGOpt;
          } else if (PGOpt) {
            // This must be a grouped option... handle them now.
            assert(isGrouping(PGOpt) && "Broken getOptionPred!");

            do {
              // Move current arg name out of RealName into RealArgName...
              std::string RealArgName(RealName.begin(),
                                      RealName.begin() + Length);
              RealName.erase(RealName.begin(), RealName.begin() + Length);

              // Because ValueRequired is an invalid flag for grouped arguments,
              // we don't need to pass argc/argv in...
              //
              assert(PGOpt->getValueExpectedFlag() != cl::ValueRequired &&
                     "Option can not be cl::Grouping AND cl::ValueRequired!");
              int Dummy;
              ErrorParsing |= ProvideOption(PGOpt, RealArgName.c_str(),
                                            0, 0, 0, Dummy);

              // Get the next grouping option...
              PGOpt = getOptionPred(RealName, Length, isGrouping);
            } while (PGOpt && Length != RealName.size());

            Handler = PGOpt; // Ate all of the options.
          }
        }
      }
    }

    if (Handler == 0) {
      llvm_cerr << ProgramName << ": Unknown command line argument '"
                << argv[i] << "'.  Try: '" << argv[0] << " --help'\n";
      ErrorParsing = true;
      continue;
    }

    // Check to see if this option accepts a comma separated list of values.  If
    // it does, we have to split up the value into multiple values...
    if (Value && Handler->getMiscFlags() & CommaSeparated) {
      std::string Val(Value);
      std::string::size_type Pos = Val.find(',');

      while (Pos != std::string::npos) {
        // Process the portion before the comma...
        ErrorParsing |= ProvideOption(Handler, ArgName,
                                      std::string(Val.begin(),
                                                  Val.begin()+Pos).c_str(),
                                      argc, argv, i);
        // Erase the portion before the comma, AND the comma...
        Val.erase(Val.begin(), Val.begin()+Pos+1);
        Value += Pos+1;  // Increment the original value pointer as well...

        // Check for another comma...
        Pos = Val.find(',');
      }
    }

    // If this is a named positional argument, just remember that it is the
    // active one...
    if (Handler->getFormattingFlag() == cl::Positional)
      ActivePositionalArg = Handler;
    else
      ErrorParsing |= ProvideOption(Handler, ArgName, Value, argc, argv, i);
  }

  // Check and handle positional arguments now...
  if (NumPositionalRequired > PositionalVals.size()) {
    llvm_cerr << ProgramName
              << ": Not enough positional command line arguments specified!\n"
              << "Must specify at least " << NumPositionalRequired
              << " positional arguments: See: " << argv[0] << " --help\n";
    
    ErrorParsing = true;
  } else if (!HasUnlimitedPositionals
             && PositionalVals.size() > PositionalOpts.size()) {
    llvm_cerr << ProgramName
              << ": Too many positional arguments specified!\n"
              << "Can specify at most " << PositionalOpts.size()
              << " positional arguments: See: " << argv[0] << " --help\n";
    ErrorParsing = true;

  } else if (ConsumeAfterOpt == 0) {
    // Positional args have already been handled if ConsumeAfter is specified...
    unsigned ValNo = 0, NumVals = PositionalVals.size();
    for (unsigned i = 0, e = PositionalOpts.size(); i != e; ++i) {
      if (RequiresValue(PositionalOpts[i])) {
        ProvidePositionalOption(PositionalOpts[i], PositionalVals[ValNo].first,
                                PositionalVals[ValNo].second);
        ValNo++;
        --NumPositionalRequired;  // We fulfilled our duty...
      }

      // If we _can_ give this option more arguments, do so now, as long as we
      // do not give it values that others need.  'Done' controls whether the
      // option even _WANTS_ any more.
      //
      bool Done = PositionalOpts[i]->getNumOccurrencesFlag() == cl::Required;
      while (NumVals-ValNo > NumPositionalRequired && !Done) {
        switch (PositionalOpts[i]->getNumOccurrencesFlag()) {
        case cl::Optional:
          Done = true;          // Optional arguments want _at most_ one value
          // FALL THROUGH
        case cl::ZeroOrMore:    // Zero or more will take all they can get...
        case cl::OneOrMore:     // One or more will take all they can get...
          ProvidePositionalOption(PositionalOpts[i],
                                  PositionalVals[ValNo].first,
                                  PositionalVals[ValNo].second);
          ValNo++;
          break;
        default:
          assert(0 && "Internal error, unexpected NumOccurrences flag in "
                 "positional argument processing!");
        }
      }
    }
  } else {
    assert(ConsumeAfterOpt && NumPositionalRequired <= PositionalVals.size());
    unsigned ValNo = 0;
    for (unsigned j = 1, e = PositionalOpts.size(); j != e; ++j)
      if (RequiresValue(PositionalOpts[j])) {
        ErrorParsing |= ProvidePositionalOption(PositionalOpts[j],
                                                PositionalVals[ValNo].first,
                                                PositionalVals[ValNo].second);
        ValNo++;
      }

    // Handle the case where there is just one positional option, and it's
    // optional.  In this case, we want to give JUST THE FIRST option to the
    // positional option and keep the rest for the consume after.  The above
    // loop would have assigned no values to positional options in this case.
    //
    if (PositionalOpts.size() == 2 && ValNo == 0 && !PositionalVals.empty()) {
      ErrorParsing |= ProvidePositionalOption(PositionalOpts[1],
                                              PositionalVals[ValNo].first,
                                              PositionalVals[ValNo].second);
      ValNo++;
    }

    // Handle over all of the rest of the arguments to the
    // cl::ConsumeAfter command line option...
    for (; ValNo != PositionalVals.size(); ++ValNo)
      ErrorParsing |= ProvidePositionalOption(ConsumeAfterOpt,
                                              PositionalVals[ValNo].first,
                                              PositionalVals[ValNo].second);
  }

  // Loop over args and make sure all required args are specified!
  for (std::map<std::string, Option*>::iterator I = Opts.begin(),
         E = Opts.end(); I != E; ++I) {
    switch (I->second->getNumOccurrencesFlag()) {
    case Required:
    case OneOrMore:
      if (I->second->getNumOccurrences() == 0) {
        I->second->error(" must be specified at least once!");
        ErrorParsing = true;
      }
      // Fall through
    default:
      break;
    }
  }

  // Free all of the memory allocated to the map.  Command line options may only
  // be processed once!
  Opts.clear();
  PositionalOpts.clear();
  MoreHelp->clear();

  // If we had an error processing our arguments, don't let the program execute
  if (ErrorParsing) exit(1);
}

//===----------------------------------------------------------------------===//
// Option Base class implementation
//

bool Option::error(std::string Message, const char *ArgName) {
  if (ArgName == 0) ArgName = ArgStr;
  if (ArgName[0] == 0)
    llvm_cerr << HelpStr;  // Be nice for positional arguments
  else
    llvm_cerr << ProgramName << ": for the -" << ArgName;
  
  llvm_cerr << " option: " << Message << "\n";
  return true;
}

bool Option::addOccurrence(unsigned pos, const char *ArgName,
                           const std::string &Value) {
  NumOccurrences++;   // Increment the number of times we have been seen

  switch (getNumOccurrencesFlag()) {
  case Optional:
    if (NumOccurrences > 1)
      return error(": may only occur zero or one times!", ArgName);
    break;
  case Required:
    if (NumOccurrences > 1)
      return error(": must occur exactly one time!", ArgName);
    // Fall through
  case OneOrMore:
  case ZeroOrMore:
  case ConsumeAfter: break;
  default: return error(": bad num occurrences flag value!");
  }

  return handleOccurrence(pos, ArgName, Value);
}

// addArgument - Tell the system that this Option subclass will handle all
// occurrences of -ArgStr on the command line.
//
void Option::addArgument(const char *ArgStr) {
  if (ArgStr[0])
    AddArgument(ArgStr, this);

  if (getFormattingFlag() == Positional)
    PositionalOptions->push_back(this);
  else if (getNumOccurrencesFlag() == ConsumeAfter) {
    if (!PositionalOptions->empty() &&
        PositionalOptions->front()->getNumOccurrencesFlag() == ConsumeAfter)
      error("Cannot specify more than one option with cl::ConsumeAfter!");
    PositionalOptions->insert(PositionalOptions->begin(), this);
  }
}

void Option::removeArgument(const char *ArgStr) {
  if (ArgStr[0])
    RemoveArgument(ArgStr, this);

  if (getFormattingFlag() == Positional) {
    std::vector<Option*>::iterator I =
      std::find(PositionalOptions->begin(), PositionalOptions->end(), this);
    assert(I != PositionalOptions->end() && "Arg not registered!");
    PositionalOptions->erase(I);
  } else if (getNumOccurrencesFlag() == ConsumeAfter) {
    assert(!PositionalOptions->empty() && (*PositionalOptions)[0] == this &&
           "Arg not registered correctly!");
    PositionalOptions->erase(PositionalOptions->begin());
  }
}


// getValueStr - Get the value description string, using "DefaultMsg" if nothing
// has been specified yet.
//
static const char *getValueStr(const Option &O, const char *DefaultMsg) {
  if (O.ValueStr[0] == 0) return DefaultMsg;
  return O.ValueStr;
}

//===----------------------------------------------------------------------===//
// cl::alias class implementation
//

// Return the width of the option tag for printing...
unsigned alias::getOptionWidth() const {
  return std::strlen(ArgStr)+6;
}

// Print out the option for the alias.
void alias::printOptionInfo(unsigned GlobalWidth) const {
  unsigned L = std::strlen(ArgStr);
  llvm_cout << "  -" << ArgStr << std::string(GlobalWidth-L-6, ' ') << " - "
            << HelpStr << "\n";
}



//===----------------------------------------------------------------------===//
// Parser Implementation code...
//

// basic_parser implementation
//

// Return the width of the option tag for printing...
unsigned basic_parser_impl::getOptionWidth(const Option &O) const {
  unsigned Len = std::strlen(O.ArgStr);
  if (const char *ValName = getValueName())
    Len += std::strlen(getValueStr(O, ValName))+3;

  return Len + 6;
}

// printOptionInfo - Print out information about this option.  The
// to-be-maintained width is specified.
//
void basic_parser_impl::printOptionInfo(const Option &O,
                                        unsigned GlobalWidth) const {
  llvm_cout << "  -" << O.ArgStr;

  if (const char *ValName = getValueName())
    llvm_cout << "=<" << getValueStr(O, ValName) << ">";

  llvm_cout << std::string(GlobalWidth-getOptionWidth(O), ' ') << " - "
            << O.HelpStr << "\n";
}




// parser<bool> implementation
//
bool parser<bool>::parse(Option &O, const char *ArgName,
                         const std::string &Arg, bool &Value) {
  if (Arg == "" || Arg == "true" || Arg == "TRUE" || Arg == "True" ||
      Arg == "1") {
    Value = true;
  } else if (Arg == "false" || Arg == "FALSE" || Arg == "False" || Arg == "0") {
    Value = false;
  } else {
    return O.error(": '" + Arg +
                   "' is invalid value for boolean argument! Try 0 or 1");
  }
  return false;
}

// parser<int> implementation
//
bool parser<int>::parse(Option &O, const char *ArgName,
                        const std::string &Arg, int &Value) {
  char *End;
  Value = (int)strtol(Arg.c_str(), &End, 0);
  if (*End != 0)
    return O.error(": '" + Arg + "' value invalid for integer argument!");
  return false;
}

// parser<unsigned> implementation
//
bool parser<unsigned>::parse(Option &O, const char *ArgName,
                             const std::string &Arg, unsigned &Value) {
  char *End;
  errno = 0;
  unsigned long V = strtoul(Arg.c_str(), &End, 0);
  Value = (unsigned)V;
  if (((V == ULONG_MAX) && (errno == ERANGE))
      || (*End != 0)
      || (Value != V))
    return O.error(": '" + Arg + "' value invalid for uint argument!");
  return false;
}

// parser<double>/parser<float> implementation
//
static bool parseDouble(Option &O, const std::string &Arg, double &Value) {
  const char *ArgStart = Arg.c_str();
  char *End;
  Value = strtod(ArgStart, &End);
  if (*End != 0)
    return O.error(": '" +Arg+ "' value invalid for floating point argument!");
  return false;
}

bool parser<double>::parse(Option &O, const char *AN,
                           const std::string &Arg, double &Val) {
  return parseDouble(O, Arg, Val);
}

bool parser<float>::parse(Option &O, const char *AN,
                          const std::string &Arg, float &Val) {
  double dVal;
  if (parseDouble(O, Arg, dVal))
    return true;
  Val = (float)dVal;
  return false;
}



// generic_parser_base implementation
//

// findOption - Return the option number corresponding to the specified
// argument string.  If the option is not found, getNumOptions() is returned.
//
unsigned generic_parser_base::findOption(const char *Name) {
  unsigned i = 0, e = getNumOptions();
  std::string N(Name);

  while (i != e)
    if (getOption(i) == N)
      return i;
    else
      ++i;
  return e;
}


// Return the width of the option tag for printing...
unsigned generic_parser_base::getOptionWidth(const Option &O) const {
  if (O.hasArgStr()) {
    unsigned Size = std::strlen(O.ArgStr)+6;
    for (unsigned i = 0, e = getNumOptions(); i != e; ++i)
      Size = std::max(Size, (unsigned)std::strlen(getOption(i))+8);
    return Size;
  } else {
    unsigned BaseSize = 0;
    for (unsigned i = 0, e = getNumOptions(); i != e; ++i)
      BaseSize = std::max(BaseSize, (unsigned)std::strlen(getOption(i))+8);
    return BaseSize;
  }
}

// printOptionInfo - Print out information about this option.  The
// to-be-maintained width is specified.
//
void generic_parser_base::printOptionInfo(const Option &O,
                                          unsigned GlobalWidth) const {
  if (O.hasArgStr()) {
    unsigned L = std::strlen(O.ArgStr);
    llvm_cout << "  -" << O.ArgStr << std::string(GlobalWidth-L-6, ' ')
              << " - " << O.HelpStr << "\n";

    for (unsigned i = 0, e = getNumOptions(); i != e; ++i) {
      unsigned NumSpaces = GlobalWidth-strlen(getOption(i))-8;
      llvm_cout << "    =" << getOption(i) << std::string(NumSpaces, ' ')
                << " - " << getDescription(i) << "\n";
    }
  } else {
    if (O.HelpStr[0])
      llvm_cout << "  " << O.HelpStr << "\n";
    for (unsigned i = 0, e = getNumOptions(); i != e; ++i) {
      unsigned L = std::strlen(getOption(i));
      llvm_cout << "    -" << getOption(i) << std::string(GlobalWidth-L-8, ' ')
                << " - " << getDescription(i) << "\n";
    }
  }
}


//===----------------------------------------------------------------------===//
// --help and --help-hidden option implementation
//

namespace {

class HelpPrinter {
  unsigned MaxArgLen;
  const Option *EmptyArg;
  const bool ShowHidden;

  // isHidden/isReallyHidden - Predicates to be used to filter down arg lists.
  inline static bool isHidden(std::pair<std::string, Option *> &OptPair) {
    return OptPair.second->getOptionHiddenFlag() >= Hidden;
  }
  inline static bool isReallyHidden(std::pair<std::string, Option *> &OptPair) {
    return OptPair.second->getOptionHiddenFlag() == ReallyHidden;
  }

public:
  HelpPrinter(bool showHidden) : ShowHidden(showHidden) {
    EmptyArg = 0;
  }

  void operator=(bool Value) {
    if (Value == false) return;

    // Copy Options into a vector so we can sort them as we like...
    std::vector<std::pair<std::string, Option*> > Opts;
    copy(Options->begin(), Options->end(), std::back_inserter(Opts));

    // Eliminate Hidden or ReallyHidden arguments, depending on ShowHidden
    Opts.erase(std::remove_if(Opts.begin(), Opts.end(),
                          std::ptr_fun(ShowHidden ? isReallyHidden : isHidden)),
               Opts.end());

    // Eliminate duplicate entries in table (from enum flags options, f.e.)
    {  // Give OptionSet a scope
      std::set<Option*> OptionSet;
      for (unsigned i = 0; i != Opts.size(); ++i)
        if (OptionSet.count(Opts[i].second) == 0)
          OptionSet.insert(Opts[i].second);   // Add new entry to set
        else
          Opts.erase(Opts.begin()+i--);    // Erase duplicate
    }

    if (ProgramOverview)
      llvm_cout << "OVERVIEW:" << ProgramOverview << "\n";

    llvm_cout << "USAGE: " << ProgramName << " [options]";

    // Print out the positional options.
    std::vector<Option*> &PosOpts = *PositionalOptions;
    Option *CAOpt = 0;   // The cl::ConsumeAfter option, if it exists...
    if (!PosOpts.empty() && PosOpts[0]->getNumOccurrencesFlag() == ConsumeAfter)
      CAOpt = PosOpts[0];

    for (unsigned i = CAOpt != 0, e = PosOpts.size(); i != e; ++i) {
      if (PosOpts[i]->ArgStr[0])
        llvm_cout << " --" << PosOpts[i]->ArgStr;
      llvm_cout << " " << PosOpts[i]->HelpStr;
    }

    // Print the consume after option info if it exists...
    if (CAOpt) llvm_cout << " " << CAOpt->HelpStr;

    llvm_cout << "\n\n";

    // Compute the maximum argument length...
    MaxArgLen = 0;
    for (unsigned i = 0, e = Opts.size(); i != e; ++i)
      MaxArgLen = std::max(MaxArgLen, Opts[i].second->getOptionWidth());

    llvm_cout << "OPTIONS:\n";
    for (unsigned i = 0, e = Opts.size(); i != e; ++i)
      Opts[i].second->printOptionInfo(MaxArgLen);

    // Print any extra help the user has declared.
    for (std::vector<const char *>::iterator I = MoreHelp->begin(),
          E = MoreHelp->end(); I != E; ++I)
      llvm_cout << *I;
    MoreHelp->clear();

    // Halt the program since help information was printed
    Options->clear();  // Don't bother making option dtors remove from map.
    exit(1);
  }
};
} // End anonymous namespace

// Define the two HelpPrinter instances that are used to print out help, or
// help-hidden...
//
static HelpPrinter NormalPrinter(false);
static HelpPrinter HiddenPrinter(true);

static cl::opt<HelpPrinter, true, parser<bool> >
HOp("help", cl::desc("Display available options (--help-hidden for more)"),
    cl::location(NormalPrinter), cl::ValueDisallowed);

static cl::opt<HelpPrinter, true, parser<bool> >
HHOp("help-hidden", cl::desc("Display all available options"),
     cl::location(HiddenPrinter), cl::Hidden, cl::ValueDisallowed);

static void (*OverrideVersionPrinter)() = 0;

namespace {
class VersionPrinter {
public:
  void operator=(bool OptionWasSpecified) {
    if (OptionWasSpecified) {
      if (OverrideVersionPrinter == 0) {
        llvm_cout << "Low Level Virtual Machine (http://llvm.org/):\n";
        llvm_cout << "  " << PACKAGE_NAME << " version " << PACKAGE_VERSION;
#ifdef LLVM_VERSION_INFO
        llvm_cout << LLVM_VERSION_INFO;
#endif
        llvm_cout << "\n  ";
#ifndef __OPTIMIZE__
        llvm_cout << "DEBUG build";
#else
        llvm_cout << "Optimized build";
#endif
#ifndef NDEBUG
        llvm_cout << " with assertions";
#endif
        llvm_cout << ".\n";
        Options->clear();  // Don't bother making option dtors remove from map.
        exit(1);
      } else {
        (*OverrideVersionPrinter)();
        exit(1);
      }
    }
  }
};
} // End anonymous namespace


// Define the --version option that prints out the LLVM version for the tool
static VersionPrinter VersionPrinterInstance;

static cl::opt<VersionPrinter, true, parser<bool> >
VersOp("version", cl::desc("Display the version of this program"),
    cl::location(VersionPrinterInstance), cl::ValueDisallowed);

// Utility function for printing the help message.
void cl::PrintHelpMessage() {
  // This looks weird, but it actually prints the help message. The
  // NormalPrinter variable is a HelpPrinter and the help gets printed when
  // its operator= is invoked. That's because the "normal" usages of the
  // help printer is to be assigned true/false depending on whether the
  // --help option was given or not. Since we're circumventing that we have
  // to make it look like --help was given, so we assign true.
  NormalPrinter = true;
}

void cl::SetVersionPrinter(void (*func)()) {
  OverrideVersionPrinter = func;
}
