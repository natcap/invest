# This finds and removes cached workspaces that have not been touched in 7 days.
# This is configured to run as a cron job in playbook.yml

base="${1%/}"  # trim a trailing slash
find "$base/twitter/local/" -maxdepth 1 -mindepth 1 -type d -mtime +7 \
  -regextype posix-egrep \
  -regex "$base/twitter/local/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$" \
  -exec rm -r {} \;
find "$base/flickr/local/" -maxdepth 1 -mindepth 1 -type d -mtime +7 \
  -regextype posix-egrep \
  -regex "$base/flickr/local/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$" \
  -exec rm -r {} \;