# Setting this up to run on cron:
# make sure this script is executeable
# sudo crontab -e
# Enter: @daily /usr/local/recreation-server/find_remove_cached_workspaces.sh

# The following finds workspace directories that have not been modified in the past 7 days, and removes them.

# Right now we have two cache dirs going, for two rec servers.
find ~/server/flickr/local/ -maxdepth 1 -mindepth 1 -type d -mtime +7 -regextype posix-egrep -regex "\/home\/davemfish\/server\/flickr\/local\/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$" -exec rm -r {} \;
find ~/server/twitter/local/ -maxdepth 1 -mindepth 1 -type d -mtime +7 -regextype posix-egrep -regex "\/home\/davemfish\/server\/twitter\/local\/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$" -exec rm -r {} \;
