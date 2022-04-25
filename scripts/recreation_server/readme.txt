To start the rec server:  sudo ./launch_recserver.sh
****************************************************
If the server is already running, the port won't be available, and sometimes there are some
zombie processes leftover after a crash. It can be useful to `sudo killall python` before launching.
See the commands in the shell script for more details, like the name of the logfile.

-----------------
DF - Sep 21, 2020

installed natcap.invest from github.com/natcap/invest:release/3.9 to get the latest server
bugfixes related to issue #304.

--------------
DF - Sep, 2020

For a while we had two servers up, one running python3 and one running python27 for compatibility
with old invest clients. That's why there are two cache directories. For a few months we have
only had the python3 server running and no one has complained.

-----------------
DF - Oct 17, 2019

Updated the invest-davemfish/invest-env-py36 environment to branch feature/INVEST-3923-Migrate-to-Python37
The recmodel_server code running here is cross-compatible with python36, so I didn't bother creating a
new 3.7 env.

-----------------
DF - July 8, 2019

We're doing python 3 updates. 
./invest-davemfish is a src tree with branch bugfix/PYTHON3-3895-27to36-compatibility
./invest-davemfish/invest-env-py36/ is a python3 env created with conda with the above branch installed.
('conda' should be available to all users. 
e.g. conda activate /usr/local/recreation_server/invest-davemfish/invest-env-py36

We launched a rec server from that environment on port 54322.
It will be live alongside the python 2 server that's already on port 54321
They share data including the input CSV table and the recserver_cache_2017/

The invest bugfix/PYTHON3-3895-27to36-compatibility client source defaults to port 54322
Once this branch has been merged and released, we can kill the python 2 server on 54321.

The new port also required:
data.naturalcapitalproject.org/server_registry/invest_recreation_model_py36/index.html

----------------
DF - July 2 2018

I'm building the cache from the 2005-2017 table. I used launch_2017.sh with recserver_cache_2017 as the cache directory, and 55555 as the port.
The idea is to build the cache/quadtree without disrupting the existing server running on port 54321. 
Then after the new cache is built: 

* kill the 'dummy' server on 55555 
* kill the 'real' server on 54321
* rename the 2017 cache dir back to recserver_cache 
* relaunch a server with the 2005-2017 table on port 54321.
        code should recognize that the quadtree already exists and skip that long process.