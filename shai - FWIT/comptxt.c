#include <stdio.h>
#include <stdbool.h>

int main(int argc, char *argv[]) {
    FILE *f1, *f2;
    char c1, c2;
    int cnt = 0, l1 = 1, l2 = 1, p1 = 1, p2 = 1;
    bool flag = true;
    

    if (argc == 4) {
        argc--;
        flag = false;
    }
    if (argc != 3) {
        printf("Invalid Input!\n");
        return 1;
    }
    f1 = fopen(argv[1], "r"), f2 = fopen(argv[2], "r");
    if (!f1 | !f2) {
        printf("File%s %s%s%s not found.\n", !f1 & !f2 ? "s" : "", !f1 ? argv[1] : "", !f1 & !f2 ? " and " : "", !f2 ? argv[2] : "");
        return 1;
    }
    for(c1 = getc(f1), c2 = getc(f2); c1 != EOF && c2 != EOF; c1 = getc(f1), c2 = getc(f2)) {
        if (c1 != c2) {
            cnt++;
            if (flag) 
                printf("Diffrence found: %c vs. %c (pos. %ld, %d:%d / %d:%d)\n", c1, c2, ftell(f1), l1, p1, l2, p2);
        }
        p1 = (c1 == '\n' ? ++l1, 1 : ++p1);
        p2 = (c2 == '\n' ? ++l2, 1 : ++p2);
    }

    if (!cnt)
        printf("The files are equal.\n");
    else if (c1 != EOF || c2 != EOF)
        printf("%d diffrence%s found, not including %s's additional length.\n", cnt, cnt == 1 ? "" : "s", c1 == EOF ? argv[2] : argv[1]);
    else
        printf("%d diffrence%s found.\n", cnt, cnt == 1 ? "" : "s");
    return 0;
}