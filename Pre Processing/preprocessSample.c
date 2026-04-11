#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <dirent.h>

#define MAX_TEXT    65536
#define MAX_TOKENS  4096
#define MAX_TOKEN   256

static const char *STOPWORDS[] = {
    "a","an","the","is","it","its","in","on","at","to","of","and","or","but",
    "for","nor","so","yet","both","either","neither","not","only","own","same",
    "than","too","very","just","because","as","until","while","although",
    "this","that","these","those","am","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should","may",
    "might","must","shall","can","about","after","before","between","into",
    "through","during","including","without","within","along","following",
    "across","behind","beyond","plus","except","up","out","around","down",
    "off","above","below","since","per","over","under","then","than","there",
    "said","says","say","also","year","years","quarter","quarters","month",
    "reported","report","reports","company","companies","corp","inc","ltd",
    "share","shares","stock","stocks","market","trading","trader","traders",
    "analyst","analysts","expect","expected","expects","according","one","two",
    "three","four","five","six","seven","eight","nine","ten","cts","dlrs",
    NULL
};

/* ---------- SGML EXTRACTION ---------- */

int extract_tag(const char *src, const char *tag, char *out, int out_max) {
    char open_tag[64], close_tag[64];
    snprintf(open_tag, sizeof(open_tag), "<%s>", tag);
    snprintf(close_tag, sizeof(close_tag), "</%s>", tag);

    const char *start = strstr(src, open_tag);
    if (!start) return 0;
    start += strlen(open_tag);

    const char *end = strstr(start, close_tag);
    if (!end) end = src + strlen(src);

    int len = (int)(end - start);
    if (len >= out_max) len = out_max - 1;

    strncpy(out, start, len);
    out[len] = '\0';
    return 1;
}

/* ---------- TAG CLEANING ---------- */

void strip_sgml_tags(const char *src, char *dst, int dst_max) {
    int inside = 0, j = 0;
    for (int i = 0; src[i] && j < dst_max-1; i++) {
        if (src[i] == '<') { inside = 1; continue; }
        if (src[i] == '>') { inside = 0; dst[j++] = ' '; continue; }
        if (!inside) dst[j++] = src[i];
    }
    dst[j] = '\0';
}

/* ---------- ENTITY NORMALIZATION ---------- */

void normalize_entity(char *token) {
    if (token[0] == '$') { strcpy(token, "AMOUNT"); return; }

    int len = strlen(token);

    if (len > 1 && token[len-1] == '%') { strcpy(token, "PERCENT"); return; }

    if (len == 2 && token[0]=='Q' && token[1]>='1' && token[1]<='4') {
        strcpy(token, "FISCALQUARTER"); return;
    }

    if (len > 1 && (token[len-1]=='B'||token[len-1]=='M'||token[len-1]=='K')) {
        int ok = 1;
        for (int i=0;i<len-1;i++)
            if (!isdigit(token[i]) && token[i]!='.') ok=0;
        if (ok) { strcpy(token,"AMOUNT"); return; }
    }

    if (len>=2 && len<=5) {
        int upper=1;
        for (int i=0;i<len;i++)
            if (!isupper(token[i])) upper=0;
        if (upper) { strcpy(token,"TICKER"); return; }
    }

    if (strchr(token,'-')) {
        int digit=0;
        for (int i=0;i<len;i++) if (isdigit(token[i])) digit=1;
        if (digit) { strcpy(token,"TIMERANGE"); return; }
    }
}

/* ---------- STOPWORD ---------- */

int is_stopword(const char *token) {
    for (int i=0; STOPWORDS[i]; i++)
        if (strcmp(token, STOPWORDS[i]) == 0) return 1;
    return 0;
}

/* ---------- STEMMER ---------- */

static int ends(char *w, int len, const char *s) {
    int slen = strlen(s);
    if (slen > len) return 0;
    return strncmp(w + len - slen, s, slen) == 0;
}

char *porter_stem(char *word) {
    int len = strlen(word);
    if (len <= 2) return word;

    if (ends(word,len,"sses")) { word[len-2]='\0'; }
    else if (ends(word,len,"ies")) { word[len-2]='\0'; }
    else if (ends(word,len,"s")) { word[len-1]='\0'; }

    return word;
}

/* ---------- TOKENIZATION PIPELINE ---------- */

int tokenize(const char *text, char token_array[][MAX_TOKEN], int max_tokens) {
    int count = 0;
    int i = 0;
    int tlen = strlen(text);

    printf("\n[STEP] RAW INPUT:\n%s\n", text);

    while (i < tlen && count < max_tokens) {

        while (i < tlen && isspace(text[i])) i++;
        if (i >= tlen) break;

        int j = 0;
        char tok[MAX_TOKEN];

        while (i < tlen && j < MAX_TOKEN-1) {
            char c = text[i];
            if (isalnum(c) || c=='$' || c=='%' || c=='.' || c=='-') {
                tok[j++] = c; i++;
            } else break;
        }

        tok[j] = '\0';
        if (j == 0) { i++; continue; }

        printf("\nToken: %s", tok);

        normalize_entity(tok);
        printf(" -> After Entity: %s", tok);

        if (strcmp(tok,"AMOUNT")!=0 && strcmp(tok,"TICKER")!=0 &&
            strcmp(tok,"PERCENT")!=0 && strcmp(tok,"TIMERANGE")!=0 &&
            strcmp(tok,"FISCALQUARTER")!=0)
        {
            for (int k=0; tok[k]; k++)
                tok[k] = tolower(tok[k]);
        }
        printf(" -> Lowercase: %s", tok);

        if (is_stopword(tok)) {
            printf(" -> Removed (Stopword)");
            continue;
        }

        if (strlen(tok) < 2) continue;

        if (strcmp(tok,"AMOUNT")!=0)
            porter_stem(tok);

        printf(" -> Stemmed: %s", tok);

        strcpy(token_array[count++], tok);
    }

    return count;
}

/* ---------- MAIN PROCESS ---------- */

void process_reuters_file(const char *filename) {

    FILE *fp = fopen(filename, "r");
    if (!fp) return;

    fseek(fp,0,SEEK_END);
    long size = ftell(fp);
    rewind(fp);

    char *buf = malloc(size+1);
    fread(buf,1,size,fp);
    buf[size]='\0';
    fclose(fp);

    char title[MAX_TEXT]={0}, body[MAX_TEXT]={0};

    extract_tag(buf,"TITLE",title,MAX_TEXT);
    extract_tag(buf,"BODY",body,MAX_TEXT);

    printf("\n[STEP] EXTRACTED TITLE:\n%s\n", title);
    printf("\n[STEP] EXTRACTED BODY:\n%s\n", body);

    char clean_title[MAX_TEXT]={0}, clean_body[MAX_TEXT]={0};

    strip_sgml_tags(title,clean_title,MAX_TEXT);
    strip_sgml_tags(body,clean_body,MAX_TEXT);

    printf("\n[STEP] AFTER TAG REMOVAL:\n%s %s\n", clean_title, clean_body);

    char combined[MAX_TEXT];
    snprintf(combined,sizeof(combined),"%s %s",clean_title,clean_body);

    char tokens[MAX_TOKENS][MAX_TOKEN];
    int n = tokenize(combined,tokens,MAX_TOKENS);

    printf("\n\n[FINAL TOKENS]:\n");
    for (int i=0;i<n;i++)
        printf("%s ", tokens[i]);

    free(buf);
}

/* ---------- MAIN ---------- */

int main() {
    process_reuters_file("sample.sgm");
    return 0;
}