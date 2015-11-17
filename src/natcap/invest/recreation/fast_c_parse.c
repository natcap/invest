#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main(int argc, char *argv[])  {
    char line[256];
    FILE* fp = fopen(argv[1], "rb");

    char user_hash_path[80];
    char lat_lng_path[80];
    char timestamp_path[80];
    strcpy(user_hash_path, argv[2]);
    strcat(user_hash_path, "hash.bin");

    strcpy(lat_lng_path, argv[2]);
    strcat(lat_lng_path, "lat_lng.bin");

    strcpy(timestamp_path, argv[2]);
    strcat(timestamp_path, "timestamp.bin");

    FILE *out_lat_lng = fopen(lat_lng_path, "w");
    FILE *out_timestamp = fopen(timestamp_path, "w");
    FILE *out_user_hash = fopen(user_hash_path, "w");

    char seps[] = ",";
    char *token;
    char *owner_name;
    char *timestamp;
    char *latitude;
    char *longitude;
    double latitude_d;
    double longitude_d;
    char *accuracy;

    time_t last_time, current_time;
    double seconds;
    time(&last_time);
    long lines_read = 0;
    fgets(line, 255, fp); //clear the header
    while(fgets(line, 255, fp) != NULL)
    {
        lines_read += 1;
        time(&current_time);
        seconds = difftime(current_time, last_time);
        if (seconds >= 5.0) {
            printf("%d lines read\n", lines_read);
            last_time = current_time;
        }
        //photo_id,owner_name,date_taken,latitude,longitude,accuracy
        token = strtok (line, seps); //photo_id
        owner_name = strtok (NULL, seps); //ownder_name
        timestamp = strtok (NULL, seps); //date_taken
        latitude = strtok (NULL, seps); //latitude
        longitude = strtok (NULL, seps); //longitude
        accuracy = strtok (NULL, seps); //accuracy

        fprintf(out_user_hash, "%s\n", owner_name);
        fprintf(out_timestamp, "%s\n", timestamp);
        latitude_d = atof(latitude);
        longitude_d = atof(longitude);
        fwrite(&latitude_d, sizeof latitude_d, 1, out_lat_lng);
        fwrite(&longitude_d, sizeof longitude_d, 1, out_lat_lng);
        //printf("%s, %s, %s, %s", owner_name, timestamp, latitude, longitude);
    }
    fclose(fp);
    return 0;
}