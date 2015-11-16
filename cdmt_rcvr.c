#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <getopt.h>


void usage(void)
{
  printf("cdmt_rcvr -p <port>\n");

  return;
}

void error(const char *msg)
{
  perror(msg);
  exit(1);
}

int main(int argc, char *argv[])
{
  int sockfd,newsockfd,*skt;
  socklen_t clilen;
  char *buffer,*header,*filename;
  struct sockaddr_in serv_addr, cli_addr;
  int bytes_read,bytes_received;
  FILE **fp;
  int port=56000;
  int arg;
  int i,j,k;
  uint32_t blocksize,nfiles,filenamesize,headersize,buffersize,serialized_int;

  // Decode options
  while ((arg=getopt(argc,argv,"p:h"))!=-1) {
    switch(arg) {

    case 'p':
      port=atoi(optarg);
      break;

    case 'h':
    default:
      usage();
      return 0;
    }
  }
  if (argc==1) {
    usage();
    return 0;
  }

  // Open socket
  sockfd=socket(AF_INET,SOCK_STREAM,0);
  if (sockfd<0) 
    error("ERROR opening socket");
  bzero((char *) &serv_addr,sizeof(serv_addr));
  serv_addr.sin_family=AF_INET;
  serv_addr.sin_addr.s_addr=INADDR_ANY;
  serv_addr.sin_port=htons(port);
  if (bind(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr))<0) 
    error("ERROR on binding!");
  
  // Listen to socket
  if (listen(sockfd,5)<0)
    error("ERROR on listen");
  
  printf("receiver at port %d openened\n",port);
  
  clilen=sizeof(cli_addr);
  newsockfd=accept(sockfd,(struct sockaddr *) &cli_addr,&clilen);
  
  if (newsockfd<0) 
    error("ERROR on accept");

  printf("New connection, socket fd is %d, ip is : %s, port : %d\n",newsockfd,inet_ntoa(cli_addr.sin_addr),ntohs(cli_addr.sin_port));

  // Read information
  bytes_read=read(newsockfd,&serialized_int,sizeof(uint32_t));
  nfiles=ntohl(serialized_int);
  bytes_read=read(newsockfd,&serialized_int,sizeof(uint32_t));
  filenamesize=ntohl(serialized_int);
  bytes_read=read(newsockfd,&serialized_int,sizeof(uint32_t));
  headersize=ntohl(serialized_int);
  bytes_read=read(newsockfd,&serialized_int,sizeof(uint32_t));
  blocksize=ntohl(serialized_int);
  bytes_read=read(newsockfd,&serialized_int,sizeof(uint32_t));
  buffersize=ntohl(serialized_int);

  // Allocate sockets
  skt=(int *) malloc(sizeof(int)*nfiles);
  skt[0]=newsockfd;

  for (i=1;i<nfiles;i++) {
    skt[i]=accept(sockfd,(struct sockaddr *) &cli_addr,&clilen);
    printf("New connection, socket fd is %d, ip is : %s, port : %d\n",skt[i],inet_ntoa(cli_addr.sin_addr),ntohs(cli_addr.sin_port));
  }

  // Allocate buffer
  buffer=(char *) malloc(sizeof(char)*blocksize);
  header=(char *) malloc(sizeof(char)*headersize);
  filename=(char *) malloc(sizeof(char)*filenamesize);

  // Zero buffer
  bzero(buffer,blocksize);

  // Allocate files
  fp=(FILE **) malloc(sizeof(FILE *)*nfiles);

  // Read file names and open files
  for (i=0;i<nfiles;i++) {
    // Read filename
    bytes_read=read(skt[i],filename,filenamesize);

    // Open file
    fp[i]=fopen(filename,"w");
  }

  // Read headers
  for (i=0;i<nfiles;i++) {
    bytes_read=read(skt[i],header,headersize);
    fwrite(header,sizeof(char),bytes_read,fp[i]);
  }

  for (k=0;;k++) {
    for (i=0;i<nfiles;i++) {
      for (bytes_received=0;;) {
	bytes_read=read(skt[i],buffer,blocksize);
	if (bytes_read==0)
	  break;
	bytes_received+=bytes_read;
	fwrite(buffer,sizeof(char),bytes_read,fp[i]);
	if (bytes_received==buffersize)
	  break;
      }
    }
    if (bytes_read==0)
      break;
  }

  // Close files
  for (i=0;i<nfiles;i++)
    fclose(fp[i]);

  // close sockets
  close(newsockfd);
  for (i=0;i<nfiles;i++)
    close(skt[i]);

  printf("receiver at port %d finished\n",port);

  // Free
  free(buffer);
  free(header);
  free(filename);
  free(fp);


  printf("Exiting %d\n",port);

  return 0; 
}
