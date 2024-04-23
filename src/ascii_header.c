/***************************************************************************
 *
 *   Copyright (C) 2002-2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "ascii_header.h"

// search header for keyword and ensure that it is preceded by whitespace */
char *ascii_header_find(char *header, const char *keyword)
{
  char *key = strstr(header, keyword);

  // keyword might be the very first word in header
  while (key > header)
  {
    // fprintf (stderr, "found=%s", key);

    // if preceded by whitespace, return the found key
    if (strchr(WHITESPACE, *(key - 1)))
      break;

    // otherwise, search again, starting one byte later
    key = strstr(key + 1, keyword);
  }

  return key;
}

int ascii_header_set(char *header, const char *keyword,
                     const char *format, ...)
{
  va_list arguments;

  char value[STRLEN];
  char *eol = 0;
  char *dup = 0;
  int ret = 0;

  /* find the keyword (also the insertion point) */
  char *key = ascii_header_find(header, keyword);

  if (key)
  {
    /* if the keyword is present, find the first '#' or '\n' to follow it */
    eol = key + strcspn(key, "#\n");
  }
  else
  {
    /* if the keyword is not present, append to the end, before "DATA" */
    eol = strstr(header, "DATA\n");
    if (eol)
      /* insert in front of DATA */
      key = eol;
    else
    {
      /* insert at end of string */
      key = header + strlen(header);
    }
  }

  va_start(arguments, format);
  ret = vsnprintf(value, STRLEN, format, arguments);
  va_end(arguments);

  if (ret < 0)
  {
    perror("ascii_header_set: error snprintf\n");
    return -1;
  }

  if (eol)
    /* make a copy */
    dup = strdup(eol);

  /* %Xs dictates only a minumum string length */
  if (sprintf(key, "%-12s %-20s   ", keyword, value) < 0)
  {
    if (dup)
      free(dup);
    perror("ascii_header_set: error sprintf\n");
    return -1;
  }

  if (dup)
  {
    strcat(key, dup);
    free(dup);
  }
  else
    strcat(key, "\n");

  return 0;
}

int ascii_header_get(char *header, const char *keyword,
                     const char *format, ...)
{
  va_list arguments;

  char *value = 0;
  int ret = 0;

  /* find the keyword */
  char *key = ascii_header_find(header, keyword);
  if (!key)
    return -1;

  /* find the value after the keyword */
  value = key + strcspn(key, WHITESPACE);

  /* parse the value */
  va_start(arguments, format);
  ret = vsscanf(value, format, arguments);
  va_end(arguments);

  return ret;
}

// This function allocates memory
char *load_file_contents_as_str(const char *filename)
{
  // Open the file for reading
  FILE *f = fopen(filename, "r");
  if (f == NULL)
  {
    fprintf(stderr, "WARNING: load_file_contents_as_str: unable to open "
                    "file %s\n",
            filename);
    return NULL;
  }

  // Get the size of the file
  fseek(f, 0L, SEEK_END);
  long size = ftell(f);
  rewind(f);

  // Allocate memory in a string buffer
  char *str = (char *)malloc(size + 1);

  // Read in the file contents to the string buffer
  long nread = fread(str, 1, size, f);
  if (nread != size)
  {
    fprintf(stderr, "warning: load_file_contents_as_str: reading in "
                    "contents of %s truncated (%ld/%ld bytes read)\n",
            filename, nread, size);
  }

  // Close file
  fclose(f);

  // Put a null termination at the end
  str[size] = '\0';

  return str;
}
