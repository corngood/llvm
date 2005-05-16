#!/usr/bin/perl -w
#
# Program:  userloc.pl
#
# Synopsis: This program uses "cvs annotate" to get a summary of how many lines
#           of code the various developres are responsible for. It takes one
#           argument, the directory to process. If the argument is not specified
#           then the cwd is used. The directory must be an LLVM tree checked out
#           from cvs. 
#
# Syntax:   userloc.pl [-details|-recurse|-tag=tag|-html... <directory>...
#
# Options:
#           -details
#               Print detailed per-directory information.
#           -recurse
#               Recurse through sub directories. Without this, only the
#               specified directory is examined
#           -tag=tag
#               Use "tag" to select the revision (as per cvs -r option)
#           -html
#               Generate HTML output instead of text output

die "Usage userloc.pl [-details|-recurse|-tag=tag|-html] <directories>..." 
  if ($#ARGV < 0);

my $tag = "";
my $details = 0;
my $recurse = 0;
my $html = 0;
while ( substr($ARGV[0],0,1) eq '-' )
{
  if ($ARGV[0] eq "-details")
  {
    $details = 1 ;
  }
  elsif ($ARGV[0] eq "-recurse")
  {
    $recurse = 1;
  }
  elsif ($ARGV[0] =~ /-tag=.*/)
  {
    $tag = $ARGV[0];
    $tag =~ s#-tag=(.*)#$1#;
  }
  elsif ($ARGV[0] eq "-html")
  {
    $html = 1;
  }
  else
  {
    die "Invalid option: $ARGV[0]";
  }
  shift;
}

die "Usage userloc.pl [-details|-recurse|-tag=tag|-html] <directories>..." 
  if ($#ARGV < 0);

my %Stats;
my %StatsDetails;

sub ValidateFile
{
  my $f = $_[0];
  my $d = $_[1];

  return 0 if ( "$f" eq "configure");
  if ( $d =~ ".*autoconf.*")
  {
    return 1 if ($f eq "configure.ac");
    return 1 if ($f eq "AutoRegen.sh");
    return 0;
  }

  return 1;
}

sub GetCVSFiles
{
  my $d = $_[0];
  my $files ="";
  open STATUS, "cvs -nfz6 status $d -l 2>/dev/null |" 
    || die "Can't 'cvs status'";
  while ( defined($line = <STATUS>) )
  {
    if ( $line =~ /^File:.*/ )
    {
      chomp($line);
      $line =~ s#^File: ([A-Za-z0-9._-]*)[ \t]*Status:.*#$1#;
      $files = "$files $d/$line" if (ValidateFile($line,$d));
    }

  }
  return $files;
}

my $annotate = "cvs annotate -lf ";
if (length($tag) > 0)
{
  $annotate = $annotate . " -r " . $tag;
}

sub ScanDir 
{
  my $Dir = $_[0];
  my $files = GetCVSFiles($Dir);

  open (DATA,"$annotate $files 2>/dev/null |")
    || die "Can't read cvs annotation data";

  my %st;
  while ( defined($line = <DATA>) )
  {
    if ($line =~ /^[0-9.]*[ \t]*\(/)
    {
      $line =~ s#^[0-9.]*[ \t]*\(([a-zA-Z0-9_.-]*).*#$1#;
      chomp($line);
      $st{$line}++;
      $Stats{$line}++;
    }
  }

  $StatsDetails{$Dir} = { %st };

  close DATA;
}

sub ValidateDirectory
{
  my $d = $_[0];
  return 0 if ($d =~ /.*CVS.*/);
  return 0 if ($d =~ /.*Debug.*/);
  return 0 if ($d =~ /.*Release.*/);
  return 0 if ($d =~ /.*Profile.*/);
  return 0 if ($d =~ /.*utils\/Burg.*/);
  return 0 if ($d =~ /.*docs\/CommandGuide\/html.*/);
  return 0 if ($d =~ /.*docs\/CommandGuide\/man.*/);
  return 0 if ($d =~ /.*docs\/CommandGuide\/ps.*/);
  return 0 if ($d =~ /.*docs\/CommandGuide\/man.*/);
  return 0 if ($d =~ /.*docs\/HistoricalNotes.*/);
  return 0 if ($d =~ /.*docs\/img.*/);
  return 0 if ($d =~ /.*bzip2.*/);
  return 1 if ($d =~ /.*projects\/Stacker.*/);
  return 1 if ($d =~ /.*projects\/sample.*/);
  return 0 if ($d =~ /.*projects\/llvm-.*/);
  return 0 if ($d =~ /.*win32.*/);
  return 1;
}

my $RowCount = 0;
sub printStats
{
  my $dir = $_[0];
  my $hash = $_[1];
  my $user;
  my $total = 0;

  if ($RowCount % 10 == 0)
  {
    print " <tr><th style=\"text-align:left\">Directory</th>\n";
    foreach $user (sort keys %Stats)
    {
      print "<th style=\"text-align:right\">",$user,"</th>\n";
    }
    print "</tr>\n";
  }

  $RowCount++;

  if ($html)
    { print "<tr><td style=\"text-align:left\">",$dir,"</td>"; }
  else
    { print $dir,"\n"; }

  foreach $user (keys %{$hash}) { $total += $hash->{$user}; }

  foreach $user ( sort keys %Stats )
  {
    my $v = $hash->{$user};
    if (defined($v))
    {
      if ($html)
      {
        printf "<td style=\"text-align:right\">%d<br/>(%2.1f%%)</td>", $v,
          (100.0/$total)*$v;
      }
      else
      {
        printf "%8d (%4.1f%%): %s\n", $v, (100.0/$total)*$v, $user;
      }
    }
    elsif ($html)
    {
      print "<td style=\"text-align:right\">-&nbsp;</td>";
    }
  }
  print "</tr>\n" if ($html);
}

my @ALLDIRS = @ARGV;

if ($recurse)
{
  $Dirs = join(" ", @ARGV);
  $Dirs = `find $Dirs -type d \! -name CVS -print`;
  @ALLDIRS = split(' ',$Dirs);
}

if ($html)
{
print "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01//EN\" \"http://www.w3.org/TR/html4/strict.dtd\">\n";
print "<html>\n<head>\n";
print "  <title>LLVM LOC Based On CVS Annotation</title>\n";
print "  <link rel=\"stylesheet\" href=\"llvm.css\" type=\"text/css\"/>\n";
print "</head>\n";
print "<body><div class=\"doc_title\">LLVM LOC Based On CVS Annotation</div>\n";
print "<p>This document shows the total lines of code per user in each\n";
print "LLVM directory. Lines of code are attributed by the user that last\n";
print "committed the line. This does not necessarily reflect authorship.</p>\n";
print "<p>The following directories were skipped:</p>\n";
print "<ol>\n";
}

for $Dir (@ALLDIRS) 
{ 
  if ( -d "$Dir" && -d "$Dir/CVS" && ValidateDirectory($Dir) )
  {
    ScanDir($Dir); 
  }
  elsif ($html)
  {
    print "<li>$Dir</li>\n";
  }
}

if ($html)
{
  print "</ol>\n";
  print "<table>\n";
}

if ($details)
{
  foreach $dir (sort keys %StatsDetails)
  {
    printStats($dir,$StatsDetails{$dir});
  }
}

printStats("Total",\%Stats);


if ($html)
{
  print "</table></body></html>\n";
}

