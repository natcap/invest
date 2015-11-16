#include <string.h>
#include <stddef.h>
#include <stdio.h>

int main(int argc, char *argv[])  {
    char line[256];
    printf (argv[1]);
    FILE* fp = fopen(argv[1], "r");
    FILE* out_fp = fopen(argv[1], "w");
    char seps[] = ",";
    char *token;
    while(fgets(line, 255, fp) != NULL)
    {
        printf("line: %s\n", line);
        token = strtok (line, seps);
        while (token != NULL)
        {
            printf("%s\n", token);
            //sscanf (token, "%d", &var);
            //input[i++] = var;
            token = strtok (NULL, seps);
        }
    }
    fclose(fp);
    return 0;
}