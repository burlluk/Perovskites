#include <stdio.h>
#include <assert.h>

typedef struct Book{
char name[8];
int volume;
}Book;

int main(int argc, char** argv){
Book shelf[10];
assert((int*)&(shelf[2].volume) == (int*)(shelf + 2) + 1); //this line changes
printf("%lu\n",sizeof(Book));
return 0;
}
