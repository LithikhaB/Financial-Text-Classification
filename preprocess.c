/*
 * preprocess.c  –  Reuters-21578 Dual-Pipeline Preprocessor
 * ==========================================================
 * Simple, clean preprocessing. No entity normalisation.
 *
 * df_traditional.csv  →  lowercase + stopwords + Porter stemming   (TF-IDF / SVM)
 * df_advanced.csv     →  lowercase + light stopwords only           (BERT / embeddings)
 *
 * Compile:  gcc -O2 -o preprocess preprocess.c
 * Usage:    ./preprocess          (auto-scans ./Dataset/)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <dirent.h>

#define MAX_TEXT    65536
#define MAX_TOKENS   4096
#define MAX_TOKEN     128
#define MAX_PATH      512

#define MODE_TRADITIONAL 0
#define MODE_ADVANCED    1

/* ─── Stopword lists ──────────────────────────────────────────────────── */

static const char *SW_LIGHT[] = {
    "a","an","the","is","it","its","in","on","at","to","of","and","or","but",
    "for","nor","so","yet","both","either","neither","not","only","own",
    "than","too","very","just","because","as","until","while","although",
    "this","that","these","those","am","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","must","shall","can","about","after","before","between",
    "into","through","during","without","within","along","across","beyond",
    "plus","except","up","out","around","down","off","above","below",
    "since","per","over","under","then","there","from","with","by",
    "we","they","he","she","who","which","what","when","where","how",
    "all","each","every","no","few","more","most","other","such","some",
    "any","if","my","our","your","their","his","her","i","you","me",
    "him","us","them","whom","whose",
    "reuter","reuters","said","says","say","also","vs",
    "mln","bln","dlrs","cts","pct","shr","rev",
    NULL
};

static const char *SW_AGGRESSIVE[] = {
    "a","an","the","is","it","its","in","on","at","to","of","and","or","but",
    "for","nor","so","yet","both","either","neither","not","only","own",
    "than","too","very","just","because","as","until","while","although",
    "this","that","these","those","am","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","must","shall","can","about","after","before","between",
    "into","through","during","without","within","along","across","beyond",
    "plus","except","up","out","around","down","off","above","below",
    "since","per","over","under","then","there","from","with","by",
    "we","they","he","she","who","which","what","when","where","how",
    "all","each","every","no","few","more","most","other","such","some",
    "any","if","my","our","your","their","his","her","i","you","me",
    "him","us","them","whom","whose",
    "reuter","reuters","said","says","say","also","vs",
    "mln","bln","dlrs","cts","pct","shr","rev",
    "year","years","quarter","month","company","corp","inc","ltd","co",
    "share","shares","stock","market","analyst","analysts",
    "report","reported","reports","expect","expected","according",
    "one","two","three","four","five","six","seven","eight","nine","ten",
    "new","old","last","next","first","late","early","note","notes",
    NULL
};

/* ─── Porter Stemmer ──────────────────────────────────────────────────── */

static int ends(const char *w, int len, const char *s) {
    int sl = (int)strlen(s);
    if (sl > len) return 0;
    return strncmp(w + len - sl, s, sl) == 0;
}
static void setto(char *w, int *len, const char *s) {
    int sl = (int)strlen(s);
    memcpy(w + *len - sl, s, sl);
    w[*len] = '\0';
    *len = (int)strlen(w);
}
static int measure(const char *w, int j) {
    int n=0,i=0;
    while(i<=j&&!(w[i]=='a'||w[i]=='e'||w[i]=='i'||w[i]=='o'||w[i]=='u')) i++;
    while(i<=j){
        while(i<=j&&(w[i]=='a'||w[i]=='e'||w[i]=='i'||w[i]=='o'||w[i]=='u')) i++;
        if(i>j)break; n++;
        while(i<=j&&!(w[i]=='a'||w[i]=='e'||w[i]=='i'||w[i]=='o'||w[i]=='u')) i++;
    }
    return n;
}
static int has_vowel(const char *w, int j) {
    for(int i=0;i<=j;i++)
        if(w[i]=='a'||w[i]=='e'||w[i]=='i'||w[i]=='o'||w[i]=='u') return 1;
    return 0;
}
static void porter_stem(char *word) {
    int len=(int)strlen(word);
    if(len<=2) return;
    if     (ends(word,len,"sses")){word[len-2]='\0';len-=2;}
    else if(ends(word,len,"ies")) {word[len-2]='\0';len-=2;}
    else if(ends(word,len,"ss"))  {}
    else if(ends(word,len,"s"))   {word[len-1]='\0';len--;}
    int flag=0;
    if(ends(word,len,"eed")){if(measure(word,len-4)>0){word[len-1]='\0';len--;}}
    else if(ends(word,len,"ed")||ends(word,len,"ing")){
        int sfx=ends(word,len,"ed")?2:3;
        if(len-sfx-1>=0&&has_vowel(word,len-sfx-1)){word[len-sfx]='\0';len-=sfx;flag=1;}
    }
    if(flag){
        if(ends(word,len,"at")||ends(word,len,"bl")||ends(word,len,"iz"))
            {word[len]='e';word[len+1]='\0';len++;}
        else if(len>=2&&word[len-1]==word[len-2]&&
                word[len-1]!='l'&&word[len-1]!='s'&&word[len-1]!='z')
            {word[len-1]='\0';len--;}
    }
    if(ends(word,len,"y")&&len>=2&&has_vowel(word,len-2)) word[len-1]='i';
    static const struct{const char*f;const char*t;}s2[]={
        {"ational","ate"},{"tional","tion"},{"enci","ence"},{"anci","ance"},
        {"izer","ize"},{"abli","able"},{"alli","al"},{"entli","ent"},
        {"eli","e"},{"ousli","ous"},{"ization","ize"},{"ation","ate"},
        {"ator","ate"},{"alism","al"},{"aliti","al"},{"ousness","ous"},
        {"iviti","ive"},{"biliti","ble"},{NULL,NULL}};
    for(int i=0;s2[i].f;i++){int sl=(int)strlen(s2[i].f);
        if(ends(word,len,s2[i].f)&&measure(word,len-sl-1)>0){setto(word,&len,s2[i].t);break;}}
    static const struct{const char*f;const char*t;}s3[]={
        {"icate","ic"},{"ative",""},{"alize","al"},{"iciti","ic"},
        {"ical","ic"},{"ful",""},{"ness",""},{NULL,NULL}};
    for(int i=0;s3[i].f;i++){int sl=(int)strlen(s3[i].f);
        if(ends(word,len,s3[i].f)&&measure(word,len-sl-1)>0){setto(word,&len,s3[i].t);break;}}
    static const char*s4[]={"al","ance","ence","er","ic","able","ible","ant",
        "ement","ment","ent","ou","ism","ate","iti","ous","ive","ize",NULL};
    for(int i=0;s4[i];i++){int sl=(int)strlen(s4[i]);
        if(ends(word,len,s4[i])&&measure(word,len-sl-1)>1){word[len-sl]='\0';break;}}
    len=(int)strlen(word);
    if(ends(word,len,"e")){
        if(measure(word,len-2)>1){word[len-1]='\0';}
        else if(measure(word,len-2)==1&&len>=3){
            int j=len-2;
            int cvc=!(word[j-1]=='a'||word[j-1]=='e'||word[j-1]=='i'||word[j-1]=='o'||word[j-1]=='u')
                   && (word[j  ]=='a'||word[j  ]=='e'||word[j  ]=='i'||word[j  ]=='o'||word[j  ]=='u')
                   &&!(word[j+1]=='a'||word[j+1]=='e'||word[j+1]=='i'||word[j+1]=='o'||word[j+1]=='u')
                   &&!(word[j]=='w'||word[j]=='x'||word[j]=='y');
            (void)cvc;
            word[len-1]='\0';
        }
    }
    len=(int)strlen(word);
    if(len>=2&&ends(word,len,"ll")&&measure(word,len-2)>1) word[len-1]='\0';
}

/* ─── Stopword check ──────────────────────────────────────────────────── */
static int is_stopword(const char *tok, int mode){
    const char **list=(mode==MODE_TRADITIONAL)?SW_AGGRESSIVE:SW_LIGHT;
    for(int i=0;list[i];i++) if(strcmp(tok,list[i])==0)return 1;
    return 0;
}

/* ─── SGML helpers ────────────────────────────────────────────────────── */
static void decode_entities(char *buf){
    char *s=buf,*d=buf;
    while(*s){
        if(*s=='&'){
            if     (strncmp(s,"&lt;",  4)==0){*d++=' ';s+=4;}
            else if(strncmp(s,"&gt;",  4)==0){*d++=' ';s+=4;}
            else if(strncmp(s,"&amp;", 5)==0){*d++='&';s+=5;}
            else if(strncmp(s,"&apos;",6)==0){*d++='\'';s+=6;}
            else if(strncmp(s,"&quot;",6)==0){*d++='"';s+=6;}
            else{*d++=*s++;}
        }else{*d++=*s++;}
    }
    *d='\0';
}
static void strip_tags(const char *src,char *dst,int dmax){
    int in=0,j=0;
    for(int i=0;src[i]&&j<dmax-1;i++){
        if(src[i]=='<'){in=1;}
        else if(src[i]=='>'){in=0;if(j<dmax-2)dst[j++]=' ';}
        else if(!in) dst[j++]=src[i];
    }
    dst[j]='\0';
}
static int extract_tag(const char *src,const char *tag,char *out,int omax){
    char op[80],cl[80];
    snprintf(op,sizeof(op),"<%s>",tag);
    snprintf(cl,sizeof(cl),"</%s>",tag);
    const char *s=strstr(src,op); if(!s)return 0;
    s+=strlen(op);
    const char *e=strstr(s,cl); if(!e)e=src+strlen(src);
    int len=(int)(e-s); if(len>=omax)len=omax-1;
    memcpy(out,s,len); out[len]='\0';
    return 1;
}

/* ─── Tokenizer ───────────────────────────────────────────────────────── */
static int emit(const char *raw,char toks[][MAX_TOKEN],int cnt,int max,int mode){
    if(cnt>=max)return 0;
    int rlen=(int)strlen(raw);
    if(rlen==0||rlen>=MAX_TOKEN)return 0;
    char tok[MAX_TOKEN];
    memcpy(tok,raw,rlen+1);

    /* lowercase */
    for(int k=0;tok[k];k++) tok[k]=(char)tolower((unsigned char)tok[k]);

    /* strip trailing punctuation */
    int l=(int)strlen(tok);
    while(l>0&&(tok[l-1]=='.'||tok[l-1]==','||tok[l-1]==';'||tok[l-1]==':'||
                tok[l-1]=='!'||tok[l-1]=='?'||tok[l-1]=='\''||tok[l-1]=='"'||
                tok[l-1]==')'||tok[l-1]==']'))
        tok[--l]='\0';
    /* strip leading punctuation */
    int st=0;
    while(tok[st]&&(tok[st]=='('||tok[st]=='['||tok[st]=='\''||tok[st]=='"'))
        st++;
    if(st>0) memmove(tok,tok+st,strlen(tok+st)+1);

    /* drop all-digit/decimal tokens */
    {int only=1;
     for(int k=0;tok[k];k++)
         if(!isdigit((unsigned char)tok[k])&&tok[k]!='.'&&tok[k]!='/'&&tok[k]!='%')
             {only=0;break;}
     if(only)return 0;}

    if((int)strlen(tok)<2) return 0;
    if(is_stopword(tok,mode)) return 0;
    if(mode==MODE_TRADITIONAL) porter_stem(tok);
    if((int)strlen(tok)<2) return 0;

    int tl=(int)strlen(tok);
    memcpy(toks[cnt],tok,tl); toks[cnt][tl]='\0';
    return 1;
}

static int tokenize(const char *text,char toks[][MAX_TOKEN],int max,int mode){
    int count=0,i=0,tlen=(int)strlen(text);
    while(i<tlen&&count<max){
        while(i<tlen&&isspace((unsigned char)text[i])) i++;
        if(i>=tlen)break;
        char raw[MAX_TOKEN]; int j=0;
        while(i<tlen&&j<MAX_TOKEN-1){
            unsigned char c=(unsigned char)text[i];
            if(isalnum(c)||c=='-'||c=='.'||c=='/'||c=='%'){raw[j++]=(char)c;i++;}
            else if(c=='\''&&j>0&&i+1<tlen&&isalpha((unsigned char)text[i+1])){i++;}
            else break;
        }
        raw[j]='\0';
        if(j==0){i++;continue;}
        count+=emit(raw,toks,count,max,mode);
    }
    return count;
}

/* ─── CSV helpers ─────────────────────────────────────────────────────── */
static void csv_escape(FILE *fp,const char *s){
    fputc('"',fp);
    for(;*s;s++){if(*s=='"')fputc('"',fp);fputc(*s,fp);}
    fputc('"',fp);
}
static void toks_to_str(char t[][MAX_TOKEN],int n,char *out,int omax){
    int pos=0;
    for(int i=0;i<n&&pos<omax-2;i++){
        int tl=(int)strlen(t[i]);
        if(pos+tl+1>=omax)break;
        if(pos>0)out[pos++]=' ';
        memcpy(out+pos,t[i],tl);pos+=tl;
    }
    out[pos]='\0';
}

/* ─── Label extractor ─────────────────────────────────────────────────── */
static int extract_labels(const char *tp,char *out,int omax){
    int pos=0,cnt=0;
    const char *p=tp;
    while((p=strstr(p,"<D>"))!=NULL){
        p+=3;
        const char *e=strstr(p,"</D>"); if(!e)break;
        int ll=(int)(e-p); if(ll<=0){p=e+4;continue;}
        if(pos>0&&pos<omax-2)out[pos++]='|';
        if(pos+ll<omax-1){memcpy(out+pos,p,ll);pos+=ll;}
        cnt++;p=e+4;
    }
    out[pos]='\0';
    return cnt;
}

/* ─── File processor ──────────────────────────────────────────────────── */
static int process_file(const char *path,FILE *ftrad,FILE *fadv){
    FILE *fp=fopen(path,"rb"); if(!fp)return 0;
    fseek(fp,0,SEEK_END); long fsz=ftell(fp); rewind(fp);
    char *buf=(char*)malloc((size_t)fsz+1); if(!buf){fclose(fp);return 0;}
    size_t nr=fread(buf,1,(size_t)fsz,fp); buf[nr]='\0';
    fclose(fp);
    decode_entities(buf);

    const char *ptr=buf; int written=0;
    while((ptr=strstr(ptr,"<REUTERS"))!=NULL){
        const char *bend=strstr(ptr,"</REUTERS>"); if(!bend)break;
        bend+=10;
        int blen=(int)(bend-ptr);
        char *blk=(char*)malloc((size_t)blen+1); if(!blk){ptr=bend;continue;}
        memcpy(blk,ptr,blen); blk[blen]='\0';

        char raw_t[MAX_TEXT]={0},raw_b[MAX_TEXT]={0},raw_tp[MAX_TEXT]={0};
        extract_tag(blk,"TITLE", raw_t, MAX_TEXT);
        extract_tag(blk,"BODY",  raw_b, MAX_TEXT);
        extract_tag(blk,"TOPICS",raw_tp,MAX_TEXT);
        if(strlen(raw_tp)<3){free(blk);ptr=bend;continue;}

        char ct[MAX_TEXT]={0},cb[MAX_TEXT]={0};
        strip_tags(raw_t,ct,MAX_TEXT);
        strip_tags(raw_b,cb,MAX_TEXT);
        char *combined=(char*)malloc(MAX_TEXT*2);
        if(!combined){free(blk);ptr=bend;continue;}
        snprintf(combined,MAX_TEXT*2,"%s %s",ct,cb);

        char labels[MAX_TEXT]={0};
        if(extract_labels(raw_tp,labels,MAX_TEXT)==0)
            {free(combined);free(blk);ptr=bend;continue;}

        {static char toks[MAX_TOKENS][MAX_TOKEN];
         int n=tokenize(combined,toks,MAX_TOKENS,MODE_TRADITIONAL);
         if(n>=3){char text[MAX_TEXT]={0};toks_to_str(toks,n,text,MAX_TEXT);
                  csv_escape(ftrad,text);fputc(',',ftrad);
                  csv_escape(ftrad,labels);fputc('\n',ftrad);}}

        {static char toks[MAX_TOKENS][MAX_TOKEN];
         int n=tokenize(combined,toks,MAX_TOKENS,MODE_ADVANCED);
         if(n>=3){char text[MAX_TEXT]={0};toks_to_str(toks,n,text,MAX_TEXT);
                  csv_escape(fadv,text);fputc(',',fadv);
                  csv_escape(fadv,labels);fputc('\n',fadv);}}

        written++;
        free(combined);free(blk);ptr=bend;
    }
    free(buf);
    return written;
}

/* ─── Main ────────────────────────────────────────────────────────────── */
int main(void){
    FILE *ftrad=fopen("df_traditional.csv","w");
    FILE *fadv =fopen("df_advanced.csv","w");
    if(!ftrad||!fadv){fprintf(stderr,"ERROR: cannot create output\n");
        if(ftrad)fclose(ftrad);if(fadv)fclose(fadv);return 1;}
    fprintf(ftrad,"\"text\",\"labels\"\n");
    fprintf(fadv, "\"text\",\"labels\"\n");

    const char *dirs[]={"Dataset","dataset",".",NULL};
    DIR *dir=NULL;const char *chosen=NULL;
    for(int d=0;dirs[d];d++){dir=opendir(dirs[d]);if(dir){chosen=dirs[d];break;}}
    if(!dir){fprintf(stderr,"ERROR: no Dataset/ folder\n");
        fclose(ftrad);fclose(fadv);return 1;}

    fprintf(stderr,"Scanning: %s\n---\n",chosen);
    struct dirent *e; char fpath[MAX_PATH];
    int tf=0,tr=0;
    while((e=readdir(dir))!=NULL){
        size_t nl=strlen(e->d_name);
        if(nl<4||strcmp(e->d_name+nl-4,".sgm")!=0)continue;
        snprintf(fpath,sizeof(fpath),"%s/%s",chosen,e->d_name);
        fprintf(stderr,"  %-36s",e->d_name);fflush(stderr);
        int r=process_file(fpath,ftrad,fadv);
        fprintf(stderr," → %d rows\n",r);
        tf++;tr+=r;
    }
    closedir(dir);fclose(ftrad);fclose(fadv);
    fprintf(stderr,"---\nDone. %d files | %d rows\n  df_traditional.csv\n  df_advanced.csv\n",tf,tr);
    return 0;
}