#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(int argc, char** argv)
{

      int i,x;
            
      srand((unsigned)time(0));
      int min = 999, max = -1;
      int val;

      x = atoi(argv[1]);

      printf("%ld ", ((long)pow(2,x)));
      
      srand(time(NULL));
                    
      for (i=0; i< pow(2,x) ; i++)
      {
         //para gerar números aleatórios positivos e negativos            
         val = (rand()%10000)-5000;
         if( val < min ) min = val;
         if( val > max ) max = val;
         printf("%d ", val);//imprime o nome, e o caracter de nova linha \n para o arquivo
      }
       
      return 0;
} 
