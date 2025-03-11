# Setting this up to run on cron:
# make sure this script is executeable
# sudo crontab -e
# Enter: @daily /path/to/this/cron/script

# The following finds workspace directories that have not been modified in the past 7 days, and removes them.

# Right now we have two cache dirs going, for two rec servers.
find /usr/local/recreation-server/invest_3_15_0/server/flickr/local/ -maxdepth 1 -mindepth 1 -type d -mtime +7 -regextype posix-egrep -regex "\/usr\/local\/recreation-server\/invest_3_15_0\/server\/flickr\/local\/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$" -exec rm -r {} \;
find /usr/local/recreation-server/invest_3_15_0/server/twitter/local/ -maxdepth 1 -mindepth 1 -type d -mtime +7 -regextype posix-egrep -regex "\/usr\/local\/recreation-server\/invest_3_15_0\/server\/twitter\/local\/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$" -exec rm -r {} \;
