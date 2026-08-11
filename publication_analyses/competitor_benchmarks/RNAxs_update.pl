#!/usr/bin/perl -w 

use strict;
use Algorithm::Diff qw(LCS);
use Getopt::Std;
use warnings;

my %opts;
getopt('stueafgwlnbdxck',\%opts);
my $max_fold=-18.40;
open(SUR,">survey.log");
if(defined($opts{s}))
{
	
	if(!defined($opts{t})) {$opts{t}=0.01157; }      
	if(!defined($opts{u})) {$opts{u}=0.001002; }  
	if(!defined($opts{e})) {$opts{e}=0.4625; }  
	if(!defined($opts{a})) {$opts{a}=0.5; }  
	if(!defined($opts{f})) {$opts{f}=0.90;}   
	if(!defined($opts{g})) {$opts{g}=0.625; }  
	if(!defined($opts{w})) {$opts{w}=80;  }  
	if(!defined($opts{l})) {$opts{l}=40;  }  
	if(!defined($opts{k})) {$opts{k}=23; }  
	if(!defined($opts{b})) {$opts{b}="N" x $opts{k};}  
		print SUR"1\n";
		my $file=read_file($opts{s});
		my $seq="";
		my $head="";
	if(!defined($opts{c})){
		foreach my $line (@{$file}){
			if($line=~/>/ ){
				$line=~s/[^A-Za-z0-9_\.]//g;
				if(length($head)>0 && length($seq)>0){
					chomp($head);
					$seq=~s/\s//g;
					`mkdir $head`;
					chdir "./$head";
					my $data=compute($seq);
					$data=selectbla($data);
					produce_output($data,$seq, $head);
					chdir "..";
					$head="";
					$seq="";
				}
				$head=$line;
			}
			else{
				chomp($line);
				$seq=$seq.$line;
			}
		}
		chomp($head);
		$seq=~s/\s//g;
		`mkdir $head`;
		chdir "./$head";
		my $data=compute($seq);
		print SUR "4\n";	
		$data=selectbla($data); #select all siRNA, now $data contains all data
		if($#$data<0){
			print SUR "10";
			close(SUR);
			printf("no siRNA found with the chosen threshold\n");
			exit(0);
		}
		print SUR "5\n";	
		produce_output($data,$seq, $head);
		print SUR "6\n";
		
		chdir "..";
		`tar -czf $head.tar.gz $head`;
		`mv $head.tar.gz ./$head`;
		print SUR `pwd`;
		if(defined($opts{d})) { design_common();}
		if(defined($opts{x})) { design_uncommon();}
		close(SUR);
	}
	else{
		$file=read_file($opts{s});
		my $siRNA=read_file($opts{c});
		foreach my $line (@{$file}){
			if($line=~/>/ ){
				$line=~s/[^A-Za-z0-9_\.]//g;
				if(length($head)>0 && length($seq)>0){
					chomp($head);
					$seq=~s/\s//g;
					my $data=compute($seq);
					$data=selectbla($data);
					$data=rank_siRNA($data);
					$head="";
					$seq="";
				}
				$head=$line;
			}
			else{
				chomp($line);
				$seq=$seq.$line;
			}
		}
		chomp($head);
		$seq=~s/\s//g;
		my $data=compute($seq);
		$data=selectbla($data); #select all siRNA, now $data contains all data
		$data=rank_siRNA($data);
		print "RANK  Pos     Target_sequence      siRNA_sequence  Acc_8  Acc_16   Asy_E   Asy_S SelfFld FreeEnd \n";
		foreach my $hit (@{$data}){
			foreach my $predicted (@{$siRNA}){
				chomp($predicted);
					if($predicted eq $hit->{siRNA}){
	    					printf("%4d %4d %s %s %1.5f %1.5f %1.5f %1.5f %1.5f %1.5f \n", $hit->{rank},$hit->{pos_t},$hit->{sequence},$hit->{siRNA},$hit->{target_8},$hit->{target_16},$hit->{e_ass},$hit->{s_ass},$hit->{self_fold},$hit->{free_end});
				}
			}
		}
	}
}
	
else{
	usage();
}


sub design_common
{

  my $seq_hash;
  my $file=read_file($opts{s});
  my $seq="";
  my $head="";
  foreach my $line (@{$file}){
    next if($line=~/^$/);
    if($line=~/>/ ){
      $line=~s/[^A-Za-z0-9_\.]//g;
      if(length($head)>0 && length($seq)>0){
	chomp($head);
	$seq=~s/\s//g;
	$seq_hash->{$head}=$seq;
	$head="";
	$seq="";
      }
      $head=$line;
    }
    else{
      chomp($line);
      $seq=$seq.$line;
    }
  }
  chomp($head);
  $seq=~s/\s//g;
  $seq_hash->{$head}=$seq;
  if(scalar keys %{$seq_hash} < 1){
    print "A least two sequences are needed!\n";
    exit(0);
  }
  else{
    my @keys = keys %{$seq_hash};
    my @seq1 =split(//,$seq_hash->{$keys[0]});
    my @seq2 =split(//,$seq_hash->{$keys[1]});
    my @results = LCS(\@seq1, \@seq2);
    for(my $i=2; $i < @keys; $i++){
      @seq1 = split(//,$seq_hash->{$keys[$i]});
      @results=LCS(\@results, \@seq1);
	last unless (scalar @results > $opts{k});
    }
    my $common_sequence=join('',@results);
    my $siRNA_hash;
		if(length($common_sequence) < $opts{k}){
			print "Common sequence too short (need at least $opts{k} nts)! No common siRNA found\n";
    }
    else{
      print("RANK  Sequence            siRNA               Asy_E   Asy_S   SelfFld FreeEnd "); 
      foreach my $head (keys %{$seq_hash}){
	$siRNA_hash->{$head}=compute($seq_hash->{$head});
	$siRNA_hash->{$head}=selectbla($siRNA_hash->{$head});
	$siRNA_hash->{$head}=rank_siRNA($siRNA_hash->{$head});
	my $leng = " " x (length($head) - 4);
	print "# mRNA ",$leng,"RANK pos  Acc_8   Acc_16  # "; 
      }
      print "\n";
      my $order;
      $common_sequence=~s/U/T/g;
    LINE: for my $i ($opts{k}..length($common_sequence)){
	my $tar=substr($common_sequence,$i-$opts{k},$opts{k});
	my $temp;
	my $check;
	for(my $j=0; $j < @keys; $j++){
	  my $count=0;
	  foreach my $hit (@{$siRNA_hash->{$keys[$j]}}){
	    if($hit->{sequence} eq $tar){
	      $count++;
	      push @{$order->{$i}->{data}->{$keys[$j]}}, $hit;
	    }
	 }
	  if($count<1){
	    delete($order->{$i});
	    next LINE;
	  }
	}
	$order->{$i}->{rank}=compute_worst($order->{$i}->{data});
      }
      my @sorted_keys = sort {$order->{$a}->{rank} <=> $order->{$b}->{rank}} keys %{$order};
      foreach my $key (@sorted_keys){
	my $hit=$order->{$key}->{data}->{$keys[0]}->[0];
	printf("%5d %s %s %1.5f %1.5f %1.5f %1.5f ",$order->{$key}->{rank},  $hit->{sequence},$hit->{siRNA},$hit->{e_ass},$hit->{s_ass},$hit->{self_fold},$hit->{free_end});
	for (my $i=0; $i< $#keys +1 ; $i++){
	  my $temp = $order->{$key}->{data}->{$keys[$i]}->[0];
	  printf ("# %s %4d %4d %1.5f %1.5f # ",$keys[$i],$temp->{rank},$temp->{pos_t},$temp->{target_8},$temp->{target_16});
	}
	print "\n";	
      }
    }
  }
}

sub compute_worst
{
	my $hash=shift;
	my $worst_rank=0;
	foreach my $key (keys %{$hash}){
		foreach my $hit (@{$hash->{$key}}){
		 	$worst_rank=($hit->{rank} > $worst_rank? $hit->{rank}:$worst_rank);
		}
	}
	return $worst_rank;
}
		 

sub design_uncommon
{
  
  my $seq_hash;
  my $file=read_file($opts{s});
  my $seq="";
  my $head="";
  foreach my $line (@{$file}){
    next if($line=~/^$/);
    if($line=~/>/ ){
      $line=~s/[^A-Za-z0-9_\.]//g;
      if(length($head)>0 && length($seq)>0){
	chomp($head);
	$seq=~s/\s//g;
	$seq_hash->{$head}=$seq;
	$head="";
	$seq="";
      }
      $head=$line;
    }
    else{
      chomp($line);
      $seq=$seq.$line;
    }
  }
  chomp($head);
  $seq=~s/\s//g;
  $seq_hash->{$head}=$seq;
  my @keys = keys %{$seq_hash};
  if(scalar keys %{$seq_hash} < 1){
    print "A least two sequences are needed!\n";
    exit(0);
  }
  else{
    my $siRNA_hash;
    foreach my $head (keys %{$seq_hash}){
      $siRNA_hash->{$head}=compute($seq_hash->{$head});
      $siRNA_hash->{$head}=selectbla($siRNA_hash->{$head});
      $siRNA_hash->{$head}=rank_siRNA($siRNA_hash->{$head});
    }
    my $siRNA_hash2;
    $siRNA_hash2 = $siRNA_hash;
    my $i=0;
    my $j=0;
    my $l=0;
    my $k=0;
    my $flag;
    $flag=0;
    for($i=0; $i<@keys ; $i++){
      for($k=0; $k<@{$siRNA_hash->{$keys[$i]}};$k++){
	$flag=0;
	for($j=$i+1; $j < @keys ; $j++){
	  next if($j==$i || !defined(@{$siRNA_hash2->{$keys[$i]}}[$k]));
	  for($l=0; $l<@{$siRNA_hash->{$keys[$j]}};$l++){				
	    next if(!defined(@{$siRNA_hash2->{$keys[$j]}}[$l]) );
	    if(@{$siRNA_hash->{$keys[$i]}}[$k]->{sequence} eq @{$siRNA_hash->{$keys[$j]}}[$l]->{sequence}){
	      $flag=1;
	      @{$siRNA_hash2->{$keys[$j]}}[$l]=();
	    }
	  }
	}
	@{$siRNA_hash2->{$keys[$i]}}[$k]=() if ($flag > 0);
	$flag=0;
      }
    }
    
    for($i=0; $i < @keys; $i++){
      print "##########$keys[$i]##########\n";
      print "RANK  Pos     Target_sequence      siRNA_sequence  Acc_8  Acc_16   Asy_E   Asy_S SelfFld FreeEnd \n";
      my $flag=0;
      for($k=0; $k<@{$siRNA_hash2->{$keys[$i]}};$k++){
	if(!defined(@{$siRNA_hash2->{$keys[$i]}}[$k])){next;}
	$flag++;
	my $hit = @{$siRNA_hash2->{$keys[$i]}}[$k]; 
	printf("%4d %4d %s %s %1.5f %1.5f %1.5f %1.5f %1.5f %1.5f \n",$hit->{rank},$hit->{pos_t},$hit->{sequence},$hit->{siRNA},$hit->{target_8},$hit->{target_16},$hit->{e_ass},$hit->{s_ass},$hit->{self_fold},$hit->{free_end});
      }
      if($flag==0) {print "NO SPECIFIC siRNA FOUND FOR $keys[$i] \n";}
      print "####################\n";
    }
  }
}


sub usage{
print
"	This is the perl script used by RNAxs\n
	to predict potent siRNAs.
	The following options can be set :
	-s A file containing mRNA sequences in FASTA format.
	-t threshold on the 8 nts accessibility
	-u threshold on the 16 nts accessibility
	-e threshold on the energy assymetry
	-a threshold on the sequence assymetry
	-f threshold on the self folding energy
	-g threshold on the siRNA structure
	-w windows size to use for RNAplfold
	-l maximal base-pair distance
	-b sequence pattern (N for any nucleotide, A,T,C,U,G)
	-k length of the designed siRNA (default:23)
	-d design siRNAs against multiple targets
	-x design target specific siRNA
	-c A file containing siRNA to be validated by RNAxs
	


	For information type: perldoc -F RNAxs.pl
";
}
	
sub read_file
{
    my $ncRNA=shift;
    open(NC, $ncRNA) || die "cant opn $ncRNA\n";
    my @nc=<NC>;
    return \@nc;
}

sub get_hit_position
{
    my $seq=shift;
    $seq=~s/c/C/g;
    $seq=~s/g/G/g;
    $seq=~s/a/A/g;
    $seq=~s/(u|t|U)/T/g;
    my $results;
    my $temp_accessibility = get_accessibility("sequence_lunp");
    my $access_8=$temp_accessibility->[0];
    my $access_16=$temp_accessibility->[1];
    for my $i ($opts{k}..length($seq)){
    	my $hash;
	my $tar=substr($seq,$i-$opts{k},$opts{k});
        my $rev=revcomp($tar);
        $hash->{sequence}=$tar;
        $hash->{siRNA}=$rev;
        $hash->{pos_t}=$i;
       	#accessibility check
        my $data=[$access_8->{$i}, $access_16->{$i}];
        next unless ($data->[0]>=$opts{t} && $data->[1]>=$opts{u});
        $hash->{target_8}=$data->[0];
       	$hash->{target_16}=$data->[1];
        #assymetry check
        $data=s_assymetry($rev);
        next unless ($data>=$opts{a});
        $hash->{s_ass}=$data;
        $data=e_assymetry($rev);
        next unless ($data>$opts{e});
        $hash->{e_ass}=$data;
        #self folding check//free end check
        $data=end_accessibility($rev);
        next unless ($data->[0]>=$opts{f} && $data->[1]>=$opts{g});
        $hash->{self_fold}=$data->[0];
       	$hash->{free_end}=$data->[1];
      push @{$results},$hash;
   }
   return $results;
}

sub get_accessibility
{
  my $name=shift;
  my $hash_8;
  my $hash_16;
  open(IN,$name);
  while(<IN>){
    next if(/#/);
    my @temp=split(/\s+/,$_);
    next unless $temp[0] > 16;
    $hash_8->{$temp[0]} =$temp[8]+0.000001;
    $hash_16->{$temp[0]}=$temp[16]+0.000001;
  }
  close(IN);
  return [$hash_8,$hash_16];
}

sub end_accessibility
{
	my $seq=shift;
	my @fold=`echo $seq | RNAfold -p`;
	chomp($fold[1]);
	$fold[1]=~/([^\s]+)\s\((\s*\-*\d+\.\d+)/;
	my $structure=$1;
	my $fold_energy=(1-$2/$max_fold);
	my $ends=substr($structure,0,4).substr($structure,length($structure)-4,4);
	my $counts=0;
	while($ends=~/(\(|\))/g){
		$counts++;
	}
	return[$fold_energy, 1-$counts/8];
}
sub s_assymetry{
	my $seq=shift;
	my $G_5=substr($seq,0,2);
	my $G_3=substr($seq,length($seq)-2,2);
	my $count_G5=0;
	my $count_G3=0;
	while($G_5=~/(G|C)/ig){
		$count_G5++;
	}
	while($G_3=~/(G|C)/g){
		$count_G3++;
	}
	my $factor_G=($count_G3-$count_G5+2)/(4);
	return $factor_G;
}
sub e_assymetry{
	my $max_ass=5.70;
	my $seq=shift;
	my $as_5=substr($seq,0,3);
	my $rc_as_5=revcomp($as_5);
	my $as_3=substr($seq,length($seq)-3,3);
	my $rc_as_3=revcomp($as_3);
	`echo "$as_5\n$rc_as_5" | RNAduplex`=~/(\-*\d+\.\d+)/;
	my $e_5=$1;
	`echo "$as_3\n$rc_as_3" | RNAduplex`=~/(\-*\d+\.\d+)/;
	my $e_3=$1;
	my $factor=($e_5-$e_3)/11.60+.5;
	return $factor;	
 }
sub revcomp{
         $a=$_[0];
         $a =~ s/(A|a)/V/g;
         $a =~ s/(G|g)/W/g;
         $a =~ s/(U|u|T|t)/A/g;
         $a =~ s/(C|c)/G/g;
         $a =~ s/(V|v)/T/g;
         $a =~ s/(W|w)/C/g;
         $a= reverse $a;
         return $a;
}

sub compute
{
    print SUR "2\n";
    my $seq=shift;
    `echo ">sequence\n$seq" |RNAplfold -W $opts{w} -L $opts{l} -u 16`;
    my $results=get_hit_position($seq);
    return $results;
}
sub selectbla
{
    my $data=shift;
    my $selected=();
    my $pattern=$opts{b};
    my $range=$opts{r};	
    $pattern=~s/(N|n)/./g;
    $pattern=~s/a/A/g;
    $pattern=~s/(U|u|t)/T/g;
    $pattern=~s/c/C/g;
    $pattern=~s/g/G/g; 
    foreach my $hit (@{$data}){
    	next unless ( 
		     $hit->{e_ass}>=$opts{e} &&
		     $hit->{s_ass}>=$opts{a} &&
		     $hit->{free_end}>=$opts{g} &&
		     $hit->{self_fold}>=$opts{f} &&
		     $hit->{target_8}>=$opts{t} &&
		     $hit->{target_16}>=$opts{u} && 
		     $hit->{siRNA}=~/^$pattern$/);
	$hit->{e_ass}	 =sprintf("%.4f", $hit->{e_ass});
	$hit->{s_ass}	 =sprintf("%.4f", $hit->{s_ass});
	$hit->{free_end} =sprintf("%.4f", $hit->{free_end});
	$hit->{self_fold}=sprintf("%.4f", $hit->{self_fold});
	$hit->{target_8} =sprintf("%.4f", $hit->{target_8});
	$hit->{target_16}=sprintf("%.4f", $hit->{target_16});
	push @{$selected}, $hit;
    }
    return $selected;
}

sub produce_output{
	my $data=shift;
	my $seq=shift;
	my $head=shift;
	my $ranked=rank_siRNA($data);
	my @temp=@{$ranked};
	produce_txt(\@temp,  $head);
}

sub produce_txt
{
	my $data=shift;
	my $head=shift;
	open(OUT,">output.csv");
	    print OUT  "RANK  Pos     Target_sequence      siRNA_sequence  Acc_8  Acc_16   Asy_E   Asy_S SelfFld FreeEnd \n";
	foreach my $hit(@{$data}){
	    printf OUT ("%4d %4d %s %s %1.5f %1.5f %1.5f %1.5f %1.5f %1.5f \n",  
		     $hit->{rank},$hit->{pos_t},$hit->{sequence},$hit->{siRNA},$hit->{target_8},
	             $hit->{target_16},$hit->{e_ass},$hit->{s_ass},$hit->{self_fold},$hit->{free_end});
	}
}

sub rank_siRNA
{
        my $data=shift;
        my $access8;
        my $access16;
        my $ass_s;
        my $ass_e;
        my $free_e;
        my $self_f;
        my $extended_hash;
	my @best;
        my $number=$opts{n};
        foreach my $hit (@{$data}){
                        $extended_hash->{$hit->{pos_t}}=$hit;
        }
        foreach my $access (reverse sort {$extended_hash->{$a}->{target_8} <=> $extended_hash->{$b}->{target_8}} keys %{$extended_hash}){
        
                push @{$access8}, $access;
                $extended_hash->{$access}->{rank}=$#$access8;
        }
        foreach my $access (reverse sort {$extended_hash->{$a}->{target_16} <=> $extended_hash->{$b}->{target_16}} keys %{$extended_hash}){
                push @{$access16}, $access;
                $extended_hash->{$access}->{rank}=($#$access16<$extended_hash->{$access}->{rank}?$extended_hash->{$access}->{rank}:$#$access16);
        }
        foreach my $access (reverse sort {$extended_hash->{$a}->{e_ass} <=> $extended_hash->{$b}->{e_ass}} keys %{$extended_hash}){
                push @{$ass_e}, $access;
                $extended_hash->{$access}->{rank}=($#$ass_e<$extended_hash->{$access}->{rank}?$extended_hash->{$access}->{rank}:$#$ass_e);
        }
        foreach my $access (reverse sort {$extended_hash->{$a}->{s_ass} <=> $extended_hash->{$b}->{s_ass}} keys %{$extended_hash}){
                push @{$ass_s}, $access;
                $extended_hash->{$access}->{rank}=($#$ass_s<$extended_hash->{$access}->{rank}?$extended_hash->{$access}->{rank}:$#$ass_s);
        }
        foreach my $key ( sort {$extended_hash->{$a}->{rank} <=> $extended_hash->{$b}->{rank}} keys %{$extended_hash}){
                push @best, $extended_hash->{$key};       
        }
	return \@best;
}

=head1 NAME

RNAxs.pl - designs siRNAs based on the target site accessibility

=head1 SYNOPSIS

	RNAxs.pl -s Sequence.seq
	RNAxs.pl -s Sequence.seq -d 1
	RNAxs.pl -s Sequence.seq -x 1 
	RNAxs.pl -s Sequence.seq -c siRNA.seq
	
=head1 DESCRIPTION

RNAxs designs siRNAs for a set of FASTA formated sequences 
contained in the file I<Sequence.seq>. For each sequences the siRNAs 
are saved in a file called I<output.csv> located in the directory
named after the sequence name.

I<output.csv> has the following structure:

B<RANK  Pos     Target sequence      siRNA sequence  Acc. 8 Acc. 16   Asy E   Asy S SelfFld FreeEnd
  26 1807 CCTTTATTCTCTTCTTAAT ATTAAGAAGAGAATAAAGG 0.54500 0.24340 0.83620 1.00000 1.00000 1.00000 
  41 1207 CCCAGTACGCTAAAACAAT ATTGTTTTAGCGTACTGGG 0.59630 0.07760 0.98280 1.00000 1.00000 1.00000>	


where 

=over 


=item I<RANK> is the worst rank of the siRNA  

=item I<POS>  is the position of the target site on the target sequence

=item I<Target sequence>  is the target sequence  

=item I<siRNA sequence>  is the siRNA sequence

=item I<Acc 8> is the accessibility of the seed region 

=item I<Acc 16>  is the accessibility of the first 15 nucleotides of the target site  

=item I<Asy E> is the duplex Energy Assymetry

=item I<Asy S>  is the duplex Sequence Assymetry 

=item I<SelfFld> is the self-folding energy 

=item I<FreeEnd>  is the number of base pairs found in the last 4 nucleotides on both 5' and 3' ends of the siRNA MFE structure

=back 

The I<-d> option allows to select siRNAs which efficiently target 
all sequences contained in I<Sequence.seq>. This is useful when all homologous
of a gene, or all splice variants should be downregulated by one siRNA.
The output is returned on stdout and has the following structure:

B< Sequence            siRNA               E_asy   S_asy   SelfFld FreeEnd # mRNA RANK pos  Acc 8   Acc 16  # # mRNA RANK pos  Acc 8   Acc 16  #
 CCCCCCCCCCCCCCGCCTT AAGGCGGGGGGGGGGGGGG 0.85340 1.00000 1.00000 1.00000 # seq1 10   105  0.24150 0.26770 # # seq2 22   92   0.22210 0.24840   
 CCCCCCCCCCCCCGCCTTG CAAGGCGGGGGGGGGGGGG 0.84480 0.75000 1.00000 1.00000 # seq1 32   342  0.22210 0.24840 # # seq2 14   102  0.21000 0.25840>

The I<-x> option allows to select siRNAs that are specific to their 
sequence. This is useful when a specific splice variant should be downregulated.
The output is returned on stdout and has the same structure as the 
I<output.csv> file.  The I<-x> and I<-d> options can be used simultaneously.

The I<-c> option allows to filter siRNAs designed by other design tools (BIOPREDsi, OptiRNA, ...) 
based on the criteria used by RNAxs. The target sequence file should
contain only one sequence. The file I<siRNA.seq> should contain one siRNA per line and each siRNA should 
match the length selected with -k (default 23 nucleotides). The output is returned on stdout
and has the same structure as I<output.csv>.

The threshold on the design features (accessiblity, asymmetry, self-folding and free end)
can be set with the following command line option:

=over

=item I<-t> sets the accessibility on the seed region  (default:0.01)

=item I<-u> sets the accessibility on the 16 nts region (default:0.001)

=item I<-e> sets the energy asymmetry (default:0.5)  

=item I<-s> sets the sequence asymmetry (default:0.5)

=item I<-f> sets the selffolding energy (default:0.9) 

=item I<-g> sets the free end parameter (default:0.7)

=back

All thresholds should be set between 0 and 1, where 0 stands for no threshold and 1 stands
for the maximal possible threshold. For example a threshold of 1 on the selffolding 
parameter would only select siRNAs devoid of structure. 

=head1 AUTHOR

Hakim Tafer <htafer@tbi.univie.ac.at>

=cut

#  End of file

