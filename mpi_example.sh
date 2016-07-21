#PBS -N outputscript
# request the default queue for this job
#PBS -q default
# request a total of 256 processors for this job (32 nodes and 8 processors per node)
#PBS -l nodes=32:ppn=8
# combine PBS standard output and error files
#PBS -j oe
# mail is sent to you when the job starts and when it terminates or aborts
#PBS -m bea
# specify your email address
#PBS -M rodrigo.g.branco@gmail.com
#change to the directory where you submitted the job

#export OMP_NUM_THREADS=8

#cd $PBS_O_WORKDIR
cd ~
#include the full path to the name of your MPI program

MY_RANDOM="$RANDOM"_"$RANDOM"_"$RANDOM"
echo "MPI P32:8"

EXECUTAVEL="exec_$MY_RANDOM"
ARQUIVO="arq_$MY_RANDOM"

mpicc -o $EXECUTAVEL mpi_perumalla.c

c=20
while [ $c -le 29 ]
do
  	echo $c
        ./gerador1 $c > $ARQUIVO
        for b in `seq 1 20`; do
             mpirun --machinefile $PBS_NODEFILE -np 256 ./$EXECUTAVEL < $ARQUIVO
        done
	rm $ARQUIVO
        echo " "
        (( c=$c + 1 ))
done

rm $EXECUTAVEL

exit 0

