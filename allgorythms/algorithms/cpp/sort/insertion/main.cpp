void insertion_sort(int s[], int n)
{
    int i, j;
    for (i=1; i<n; i++) {
        j=i;
        while ((j>0) && (s[j] < s[j-1])) {
            int tmp = s[j];
            s[j]=s[j-1];
            s[j-1] = tmp;
            j=j-1;
        }
    }
}
