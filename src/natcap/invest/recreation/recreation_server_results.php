<?php
//read configuration
$json = file_get_contents("./recreation_server_config.json");
$config = json_decode($json);

//get session id
// $sessid = "DEBUGGING";
$sessid = $_POST["sessid"];

//set paths
$usrpath = $config->{'paths'}->{'absolute'}->{'userroot'} . $config->{'paths'}->{'relative'}->{'data'};
$webpath = $config->{'paths'}->{'absolute'}->{'webroot'} . $config->{'paths'}->{'relative'}->{'data'};
$sesspath = $usrpath . $sessid . "/";

$results = $config->{'files'}->{'results'};

//open log
$logpath = $sesspath . $config->{'files'}->{'PHPlog'};
$log = fopen($logpath, 'a');
fwrite($log,"Begin results.php");
fflush($log);

//add symlinks for public grid
exec("ln -s " . $sesspath . $config->{'files'}->{'grid'}->{'shp'} . " " . $sesspath . $config->{'files'}->{'grid_tmp'}->{'shp'});
exec("ln -s " . $sesspath . $config->{'files'}->{'grid'}->{'shx'} . " " . $sesspath . $config->{'files'}->{'grid_tmp'}->{'shx'});
exec("ln -s " . $sesspath . $config->{'files'}->{'grid_public'}->{'dbf'} . " " . $sesspath . $config->{'files'}->{'grid_tmp'}->{'dbf'});
exec("ln -s " . $sesspath . $config->{'files'}->{'grid'}->{'prj'} . " " . $sesspath . $config->{'files'}->{'grid_tmp'}->{'prj'});

//zip standard outputs
$command="zip -j " . $sesspath . $results . 
" " . $sesspath . $config->{'files'}->{'grid_tmp'}->{'shp'} . 
" " . $sesspath . $config->{'files'}->{'grid_tmp'}->{'shx'} .
" " . $sesspath . $config->{'files'}->{'grid_tmp'}->{'dbf'} . 
" " . $sesspath . $config->{'files'}->{'grid_tmp'}->{'prj'} .
" " . $sesspath . $config->{'files'}->{'JSON'}->{'init'} .
" " . $sesspath . $config->{'files'}->{'JSON'}->{'scenario'} .
" " . $sesspath . $config->{'files'}->{'params'} .
" " . $sesspath . $config->{'files'}->{'regression_summary'} .
" " . $sesspath . $config->{'files'}->{'comments'};
fwrite($log,"\n" . "Executing command: " . $command);
fflush($log);
exec($command);

//zip any download predictors
//I'm not thrilled about using a recursive zip on a relative path, but it seems unavoidable
chdir($sesspath);
exec("zip -r " . $results . " " . $config->{'paths'}->{'relative'}->{'download'});

//remove symlinks for public grid
exec("rm " . $sesspath . $config->{'files'}->{'grid_tmp'}->{'shp'});
exec("rm " . $sesspath . $config->{'files'}->{'grid_tmp'}->{'shx'});
exec("rm " . $sesspath . $config->{'files'}->{'grid_tmp'}->{'dbf'});
exec("rm " . $sesspath . $config->{'files'}->{'grid_tmp'}->{'prj'});

//symlink results to web path
$command="ln -s " . $sesspath . $results
. " " . $webpath . $sessid  . "/";
fwrite($log,"\n" . "Executing command: " . $command);
fflush($log);
exec($command);

echo $sessid;

//close log
fwrite($log,"\nEnd results.php\n");
fclose($log);
?>
