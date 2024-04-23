#include "read_vdif.h"

struct vdif_file *init_vdif_struct(unsigned int nsub)
{
    struct vdif_file *vf;

    // Allocate memory for the struct
    vf = (struct vdif_file *) malloc(sizeof(struct vdif_file));
    memset(vf, 0, sizeof(struct vdif_file));

    // Allocate memory for the file names
    vf->hdrfns = (char **)malloc(sizeof(char *)*nsub);
    if (vf->hdrfns==NULL) {
        fprintf(stderr, "ERROR :: Failed to allocate memory for header file names\n");
        free(vf);
        exit(1);
    }
    vf->datafns = (char **)malloc(sizeof(char *)*nsub);
    if (vf->datafns==NULL) {
        fprintf(stderr, "ERROR :: Failed to allocate memory for data file names\n");
        free(vf->hdrfns);
        free(vf);
        exit(1);
    }

    // Initialise individual strings
    for (int i=0;i<nsub;i++) {
        vf->datafns[i] = (char *)malloc(4096);
        vf->hdrfns[i] = NULL;  // TODO implement check for NULL
    }
    vf->lowdatafn = (char *)malloc(4096);
    vf->telescope = (char *)malloc(256);
    vf->src_name = (char *)malloc(256);
    vf->src_raj = (char *)malloc(32);
    vf->src_dej = (char *)malloc(32);

    return vf;
}

void free_vdif_struct(struct vdif_file *vf,unsigned int nsub)
{
    for (int i=0;i<nsub;i++) {
        free(vf->datafns[i]);
        if (vf->hdrfns && vf->hdrfns[i])
            free(vf->hdrfns[i]);
    }
    free(vf->datafns);
    free(vf->hdrfns);
    free(vf->lowdatafn);
    free(vf->telescope);
    free(vf->src_name);
    free(vf->src_raj);
    free(vf->src_dej);
    free(vf);
}

int parse_ascii_header(struct vdif_file *vf,unsigned int nsub,char *hdrfn,char *src_name)
{
    int lowest_subband;
    char *common_fn;
    char *ascii_hdr;
    char datafn[4096],datafn_path[4096];;
    double lowfreq, highfreq;
    double start_mjd;
    char *s;
    int n;

    if ((!vf)||(!hdrfn)||(!src_name)) {
        fprintf(stderr, "ERROR :: Invalid pointer passed to vdif_file struct. Exiting.\n");
        return 1;
    }

    // Initialise the header file names
    lowest_subband=get_lowest_subband(hdrfn);
    common_fn=get_common_fname(hdrfn);
    for (int i=0;i<nsub;i++) {
        vf->hdrfns[i]=(char *)malloc(strlen(hdrfn)+1);
        sprintf(vf->hdrfns[i],"%s_ch%03d.hdr",common_fn,lowest_subband+i);
    }
    free(common_fn);

    // Parse info from the ascii header
    ascii_hdr=load_file_contents_as_str(vf->hdrfns[0]);
    ascii_header_get(ascii_hdr,"TELESCOPE","%s",vf->telescope);
    ascii_header_get(ascii_hdr,"SOURCE","%s",vf->src_name);
    ascii_header_get(ascii_hdr,"RA","%s",vf->src_raj);
    ascii_header_get(ascii_hdr,"DEC","%s",vf->src_dej);
    ascii_header_get(ascii_hdr,"FREQ","%lf",&lowfreq);
    ascii_header_get(ascii_hdr,"BW","%lf",&vf->bwchan);
    ascii_header_get(ascii_hdr,"DATAFILE","%s",datafn);
    free(ascii_hdr);

    // Store the raw data file name from the lowest subband
    strcpy(vf->lowdatafn, datafn);

    // If a source name is provided, override the one from the header
    if (src_name[0] != '\0')
        strcpy(vf->src_name, src_name);

    // Now get the centre frequency from the highest subband
    ascii_hdr=load_file_contents_as_str(vf->hdrfns[nsub-1]);
    ascii_header_get(ascii_hdr,"FREQ","%lf",&highfreq);
    free(ascii_hdr);

    // Prepend the same path from hdrfile to datafile
    s=strrchr(hdrfn, '/');
    if (!s) {
        // If no path was found, don't prepend one!
        strcpy(datafn_path, datafn);
    } else {
        // Some pointer fun to copy the path, and then append the filename
        n=s-hdrfn+1;
        strncpy(datafn_path, hdrfn,s-hdrfn+1);
        s=datafn_path+n;
        strcpy(s,datafn);
    }

    // Get start time from VDIF packet header
    start_mjd=get_start_mjd(datafn_path);
    if (start_mjd==-1) {
        fprintf(stderr, "ERROR :: Unable to get start time from VDIF header. Exiting.\n");
        return 1;
    }

    // Compute observation parameters
    vf->fcen=(lowfreq+highfreq)/2.0;
    vf->tsamp=1.0/(vf->bwchan*1e6);
    vf->tstart=start_mjd;

    // Initialise the data file names/paths
    common_fn=get_common_fname(datafn_path);
    for (int i=0;i<nsub;i++)
        sprintf(vf->datafns[i],"%s_ch%03d.vdif",common_fn,lowest_subband+i);
    free(common_fn);

    // Try to open files
    for (int i=0;i<nsub;i++) {
        if (access(vf->hdrfns[i],F_OK)==-1)
        {
            fprintf(stderr, "ERROR :: Unable to access header file %s. Exiting.\n", vf->hdrfns[i]);
            return 1;
        }
        if (access(vf->datafns[i],F_OK)==-1)
        {
            fprintf(stderr, "ERROR :: Unable to access data file %s. Exiting.\n", vf->datafns[i]);
            return -1;
        }
    }

    return 0;
}

int get_lowest_subband(char *hdrfile)
{
    char *start;
    int sub = 0;

    start=strstr(hdrfile,"_ch");
    if (start)
        sub=atoi(start+3);
    return sub;
}

char *get_common_fname(char *hdrfile)
{
    char *common_fname;
    char *end;
    int len;

    end=strstr(hdrfile,"_ch");
    if (end) {
        len=end-hdrfile;
        common_fname=malloc(len+1);
        strncpy(common_fname,hdrfile,len);
        common_fname[len]='\0';
        return common_fname;
    }
    return NULL;
}

double get_start_mjd(char *datafn)
{
    FILE *file;
    struct vdif_header *hdr;
    double mjd;

    file=fopen(datafn,"r");
    if (file==NULL) {
        fprintf(stderr, "ERROR :: Unable to open file %s to read header information. Exiting.\n", datafn);
        return -1;
    }
    hdr=(struct vdif_header *)malloc(VDIF_HEADER_BYTES);
    fread(hdr,1,VDIF_HEADER_BYTES,file);
    fclose(file);
    mjd=getVDIFFrameDMJD(hdr,FPS);
    free(hdr);
    return mjd;
}

int read_block_and_strip(FILE *file,char *vfbuf,char *tmp_vfbuf,int isub,int nframe)
{
    int nread;
    int nsamp_read;
    int frames_read;
    int datalength=VDIF_DATA_BYTES;

    // Read nframe frames into a temporary buffer
    memset(tmp_vfbuf,0,VDIF_FRAME_BYTES*nframe);
    nread=fread(tmp_vfbuf,1,VDIF_FRAME_BYTES*nframe,file);

    if (nread!=VDIF_FRAME_BYTES*nframe) {
        frames_read=floor(nread/VDIF_FRAME_BYTES);
    } else {
        frames_read=nframe;
    }
    
    // Copy data into vfbuf, excluding the frame headers
    for (int iframe=0;iframe<frames_read;iframe++) {
        memcpy(vfbuf+(isub*frames_read+iframe)*datalength,\
            tmp_vfbuf+iframe*VDIF_FRAME_BYTES+VDIF_HEADER_BYTES,\
            datalength);
    }
    
    // Calculate how many time samples were read (128 per frame)
    nsamp_read=(nread-frames_read*VDIF_HEADER_BYTES)/4;
    return nsamp_read;
}