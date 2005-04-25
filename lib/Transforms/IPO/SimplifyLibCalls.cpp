//===- SimplifyLibCalls.cpp - Optimize specific well-known library calls --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a variety of small optimizations for calls to specific
// well-known (e.g. runtime library) function calls. For example, a call to the
// function "exit(3)" that occurs within the main() function can be transformed
// into a simple "return 3" instruction. Any optimization that takes this form
// (replace call to library function with simpler code that provides same 
// result) belongs in this file. 
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/hash_map"
#include <iostream>
using namespace llvm;

namespace {
  Statistic<> SimplifiedLibCalls("simplified-lib-calls", 
      "Number of well-known library calls simplified");

  /// This class is the base class for a set of small but important 
  /// optimizations of calls to well-known functions, such as those in the c
  /// library. This class provides the basic infrastructure for handling 
  /// runOnModule. Subclasses register themselves and provide two methods:
  /// RecognizeCall and OptimizeCall. Whenever this class finds a function call,
  /// it asks the subclasses to recognize the call. If it is recognized, then
  /// the OptimizeCall method is called on that subclass instance. In this way
  /// the subclasses implement the calling conditions on which they trigger and
  /// the action to perform, making it easy to add new optimizations of this
  /// form.
  /// @brief A ModulePass for optimizing well-known function calls
  struct SimplifyLibCalls : public ModulePass {


    /// For this pass, process all of the function calls in the module, calling
    /// RecognizeCall and OptimizeCall as appropriate.
    virtual bool runOnModule(Module &M);

  };

  RegisterOpt<SimplifyLibCalls> 
    X("simplify-libcalls","Simplify well-known library calls");

  struct CallOptimizer
  {
    /// @brief Constructor that registers the optimization
    CallOptimizer(const char * fname );

    virtual ~CallOptimizer();

    /// The implementation of this function in subclasses should determine if
    /// \p F is suitable for the optimization. This method is called by 
    /// runOnModule to short circuit visiting all the call sites of such a
    /// function if that function is not suitable in the first place.
    /// If the called function is suitabe, this method should return true;
    /// false, otherwise. This function should also perform any lazy 
    /// initialization that the CallOptimizer needs to do, if its to return 
    /// true. This avoids doing initialization until the optimizer is actually
    /// going to be called upon to do some optimization.
    virtual bool ValidateCalledFunction(
      const Function* F ///< The function that is the target of call sites
    ) const = 0;

    /// The implementations of this function in subclasses is the heart of the 
    /// SimplifyLibCalls algorithm. Sublcasses of this class implement 
    /// OptimizeCall to determine if (a) the conditions are right for optimizing
    /// the call and (b) to perform the optimization. If an action is taken 
    /// against ci, the subclass is responsible for returning true and ensuring
    /// that ci is erased from its parent.
    /// @param ci the call instruction under consideration
    /// @param f the function that ci calls.
    /// @brief Optimize a call, if possible.
    virtual bool OptimizeCall(
      CallInst* ci ///< The call instruction that should be optimized.
    ) const = 0;

    const char * getFunctionName() const { return func_name; }
  private:
    const char* func_name;
  };

  /// @brief The list of optimizations deriving from CallOptimizer

  hash_map<std::string,CallOptimizer*> optlist;

  CallOptimizer::CallOptimizer(const char* fname)
    : func_name(fname)
  {
    // Register this call optimizer
    optlist[func_name] = this;
  }

  /// Make sure we get our virtual table in this file.
  CallOptimizer::~CallOptimizer() 
  {
    optlist.clear();
  }
}

ModulePass *llvm::createSimplifyLibCallsPass() 
{ 
  return new SimplifyLibCalls(); 
}

bool SimplifyLibCalls::runOnModule(Module &M) 
{
  bool result = false;

  // The call optimizations can be recursive. That is, the optimization might
  // generate a call to another function which can also be optimized. This way
  // we make the CallOptimizer instances very specific to the case they handle.
  // It also means we need to keep running over the function calls in the module
  // until we don't get any more optimizations possible.
  bool found_optimization = false;
  do
  {
    found_optimization = false;
    for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI)
    {
      // All the "well-known" functions are external and have external linkage
      // because they live in a runtime library somewhere and were (probably) 
      // not compiled by LLVM.  So, we only act on external functions that have 
      // external linkage and non-empty uses.
      if (FI->isExternal() && FI->hasExternalLinkage() && !FI->use_empty())
      {
        // Get the optimization class that pertains to this function
        if (CallOptimizer* CO = optlist[FI->getName().c_str()] )
        {
          // Make sure the called function is suitable for the optimization
          if (CO->ValidateCalledFunction(FI))
          {
            // Loop over each of the uses of the function
            for (Value::use_iterator UI = FI->use_begin(), UE = FI->use_end(); 
                 UI != UE ; )
            {
              // If the use of the function is a call instruction
              if (CallInst* CI = dyn_cast<CallInst>(*UI++))
              {
                // Do the optimization on the CallOptimizer.
                if (CO->OptimizeCall(CI))
                {
                  ++SimplifiedLibCalls;
                  found_optimization = result = true;
                }
              }
            }
          }
        }
      }
    }
  } while (found_optimization);
  return result;
}

namespace {

/// This CallOptimizer will find instances of a call to "exit" that occurs
/// within the "main" function and change it to a simple "ret" instruction with
/// the same value as passed to the exit function. It assumes that the 
/// instructions after the call to exit(3) can be deleted since they are 
/// unreachable anyway.
/// @brief Replace calls to exit in main with a simple return
struct ExitInMainOptimization : public CallOptimizer
{
  ExitInMainOptimization() : CallOptimizer("exit") {}
  virtual ~ExitInMainOptimization() {}

  // Make sure the called function looks like exit (int argument, int return
  // type, external linkage, not varargs). 
  virtual bool ValidateCalledFunction(const Function* f) const
  {
    if (f->getReturnType()->getTypeID() == Type::VoidTyID && !f->isVarArg())
      if (f->arg_size() == 1)
        if (f->arg_begin()->getType()->isInteger())
          return true;
    return false;
  }

  virtual bool OptimizeCall(CallInst* ci) const
  {
    // To be careful, we check that the call to exit is coming from "main", that
    // main has external linkage, and the return type of main and the argument
    // to exit have the same type. 
    Function *from = ci->getParent()->getParent();
    if (from->hasExternalLinkage())
      if (from->getReturnType() == ci->getOperand(1)->getType())
        if (from->getName() == "main")
        {
          // Okay, time to actually do the optimization. First, get the basic 
          // block of the call instruction
          BasicBlock* bb = ci->getParent();

          // Create a return instruction that we'll replace the call with. 
          // Note that the argument of the return is the argument of the call 
          // instruction.
          ReturnInst* ri = new ReturnInst(ci->getOperand(1), ci);

          // Split the block at the call instruction which places it in a new
          // basic block.
          bb->splitBasicBlock(BasicBlock::iterator(ci));

          // The block split caused a branch instruction to be inserted into
          // the end of the original block, right after the return instruction
          // that we put there. That's not a valid block, so delete the branch
          // instruction.
          bb->back().eraseFromParent();

          // Now we can finally get rid of the call instruction which now lives
          // in the new basic block.
          ci->eraseFromParent();

          // Optimization succeeded, return true.
          return true;
        }
    // We didn't pass the criteria for this optimization so return false
    return false;
  }
} ExitInMainOptimizer;

/// This CallOptimizer will simplify a call to the strcat library function. The
/// simplification is possible only if the string being concatenated is a 
/// constant array or a constant expression that results in a constant array. In
/// this case, if the array is small, we can generate a series of inline store
/// instructions to effect the concatenation without calling strcat.
/// @brief Simplify the strcat library function.
struct StrCatOptimization : public CallOptimizer
{
  StrCatOptimization() : CallOptimizer("strcat") {}
  virtual ~StrCatOptimization() {}

  /// @brief Make sure that the "strcat" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f) const
  {
    if (f->getReturnType() == PointerType::get(Type::SByteTy))
      if (f->arg_size() == 2) 
      {
        Function::const_arg_iterator AI = f->arg_begin();
        if (AI++->getType() == PointerType::get(Type::SByteTy))
          if (AI->getType() == PointerType::get(Type::SByteTy))
            return true;
      }
    return false;
  }

  /// Perform the optimization if the length of the string concatenated
  /// is reasonably short and it is a constant array.
  virtual bool OptimizeCall(CallInst* ci) const
  {
    // If the thing being appended is not a GEP instruction
    GetElementPtrInst* GEP = dyn_cast<GetElementPtrInst>(ci->getOperand(2));
    if (!GEP)
      return false;

    // Double check that we're dealing with a pointer to sbyte here
    if (GEP->getType() != PointerType::get(Type::SByteTy))
      return false;

    // We can only optimize if the appended string is a constant 
    Constant* C = dyn_cast<Constant>(GEP->getPointerOperand());
    if (!C)
      return false;

    // Check the various kinds of constants that are applicable
    GlobalVariable* GV = dyn_cast<GlobalVariable>(C);
    if (!GV)
      return false;

    // Only GVars that have initializers will do
    if (GV->hasInitializer())
    {
      Constant* INTLZR = GV->getInitializer();
      // And only if that initializer is ConstantArray
      if (ConstantArray* A = dyn_cast<ConstantArray>(INTLZR))
      {
        assert(A->isString() && "This ought to be a string");

        // Get the value of the string and determine its length. If the length
        // is zero, we can just substitute the destination pointer for the
        // call. 
        std::string str = A->getAsString().c_str();
        if (str.length() == 0)
        {
          ci->replaceAllUsesWith(ci->getOperand(1));
          ci->eraseFromParent();
          return true;
        }

        // Otherwise, lets just turn this into a memcpy call which will be 
        // optimized out on the next pass.
        else
        {
          // Extract some information
          Module* M = ci->getParent()->getParent()->getParent();
          // We need to find the end of the string of the first operand to the
          // strcat call instruction. That's where the memory is to be moved
          // to. So, generate code that does that
          std::vector<const Type*> args;
          args.push_back(PointerType::get(Type::SByteTy));
          FunctionType* strlen_type = 
            FunctionType::get(Type::IntTy, args, false);
          Function* strlen = M->getOrInsertFunction("strlen",strlen_type);
          CallInst* strlen_inst = 
            new CallInst(strlen,ci->getOperand(1),"",ci);

          // Now that we have the string length, we must add it to the pointer
          // to get the memcpy destination.
          std::vector<Value*> idx;
          idx.push_back(strlen_inst);
          GetElementPtrInst* gep = 
            new GetElementPtrInst(ci->getOperand(1),idx,"",ci);

          // Generate the memcpy call
          args.clear();
          args.push_back(PointerType::get(Type::SByteTy));
          args.push_back(PointerType::get(Type::SByteTy));
          args.push_back(Type::IntTy);
          FunctionType* memcpy_type = FunctionType::get(
            PointerType::get(Type::SByteTy), args, false);
          Function* memcpy = M->getOrInsertFunction("memcpy",memcpy_type);
          std::vector<Value*> vals;
          vals.push_back(gep);
          vals.push_back(ci->getOperand(2));
          vals.push_back(ConstantSInt::get(Type::IntTy,str.length()+1));
          CallInst* memcpy_inst = new CallInst(memcpy, vals, "", ci);

          // Finally, cast the result of the memcpy to the correct type which is
          // the result of the strcat.
          CastInst* cast_inst =
            new CastInst(memcpy_inst, PointerType::get(Type::SByteTy),
                ci->getName(),ci);

          // And perform the stubstitution for the strcat call.
          ci->replaceAllUsesWith(cast_inst);
          ci->eraseFromParent();
          return true;
        }
      }
      else if (ConstantAggregateZero* CAZ = 
          dyn_cast<ConstantAggregateZero>(INTLZR))
      {
        // We know this is the zero length string case so we can just avoid
        // the strcat altogether. 
        ci->replaceAllUsesWith(ci->getOperand(1));
        ci->eraseFromParent();
        return true;
      }
      else if (ConstantExpr* E = dyn_cast<ConstantExpr>(INTLZR))
      {
        return false;
      }
    }

    // We didn't pass the criteria for this optimization so return false.
    return false;
  }
} StrCatOptimizer;

/// This CallOptimizer will simplify a call to the memcpy library function by
/// expanding it out to a small set of stores if the copy source is a constant
/// array. 
/// @brief Simplify the memcpy library function.
struct MemCpyOptimization : public CallOptimizer
{
  MemCpyOptimization() : CallOptimizer("memcpy") {}
  virtual ~MemCpyOptimization() {}

  /// @brief Make sure that the "memcpy" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f) const
  {
    if (f->getReturnType() == PointerType::get(Type::SByteTy))
      if (f->arg_size() == 2) 
      {
        Function::const_arg_iterator AI = f->arg_begin();
        if (AI++->getType() == PointerType::get(Type::SByteTy))
          if (AI->getType() == PointerType::get(Type::SByteTy))
            return true;
      }
    return false;
  }

  /// Perform the optimization if the length of the string concatenated
  /// is reasonably short and it is a constant array.
  virtual bool OptimizeCall(CallInst* ci) const
  {
    // We didn't pass the criteria for this optimization so return false.
    return false;
  }
} MemCpyOptimizer;
}
