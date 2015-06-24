
<?php

//read configuration
$json = file_get_contents("./recreation_server_config.json");
$config = json_decode($json);

//set paths
$usrpath = $config->{'paths'}->{'absolute'}->{'userroot'} . $config->{'paths'}->{'relative'}->{'data'};
$webpath = $config->{'paths'}->{'absolute'}->{'webroot'} . $config->{'paths'}->{'relative'}->{'data'};
$predpath = $config->{'paths'}->{'relative'}->{'predictors'};

//get session id
session_start();
$sessid = session_id();

//set session path
$sesspath = $usrpath . $sessid . "/";

//make session directories
mkdir($usrpath . $sessid, 0755);
mkdir($usrpath . $sessid . "/" . $predpath , 0755);

//open log
$logpath = $usrpath . $sessid . "/" . $config->{'files'}->{'log'};
$log = fopen($logpath, 'w');
fwrite($log,",DEBUG,Begin version PHP script.\n");
fflush($log);

//record ip address
$ipaddress = str_replace(".", "_", $_SERVER["REMOTE_ADDR"]);
fwrite($log,",DEBUG,IP address: " . $ipaddress . ".\n");
fflush($log);

//symlink session log to web path
mkdir($webpath . $sessid, 0755);
exec("ln -s " . $sesspath . $config->{'files'}->{'log'} . " "
. $webpath . $sessid  . "/");

if ($_POST["is_release"]=="True")
{
    fwrite($log,",DEBUG,Detected release version.\n");
    fflush($log);

    fwrite($log,",DEBUG,Version " . $_POST["version_info"] . ".\n");

    $v = substr($_POST["version_info"],0,1);
    $j = substr($_POST["version_info"],2,1);
    $n = substr($_POST["version_info"],4,1);

    fwrite($log,",DEBUG,Major: " . $v . ".\n");
    fwrite($log,",DEBUG,Minor: " . $j . ".\n");
    fwrite($log,",DEBUG,Build: " . $n . ".\n");

    if (($v > "2") or (($v == "2") and (($j > "5") or (($j == "5") and ($n >= "4"))))){
        fwrite($log,",INFO,You have a compatible version.\n");
        fflush($log);    
    }
    else
    {
        fwrite($log,",ERROR,Download the new version of InVEST.\n");
        fflush($log);    
    }        
}
else
{
    fwrite($log,",INFO,Developer version detected.\n");
    fflush($log);    
}

echo $sessid;

//close log
fwrite($log,",DEBUG,End version PHP script.\n");
fclose($log);
?>
