<?php
//read configuration
$json = file_get_contents("./recreation_server_config.json");
$config = json_decode($json);

//set paths
$usrpath = $config->{'paths'}->{'absolute'}->{'userroot'} . $config->{'paths'}->{'relative'}->{'data'};
$webpath = $config->{'paths'}->{'absolute'}->{'webroot'} . $config->{'paths'}->{'relative'}->{'data'};
$recpath = $config->{'paths'}->{'absolute'}->{'repository'};

//parse model parameters
$model = json_decode($_POST["json"]);

//set session id
$sessid = $model->{'sessid'};
$sesspath = $usrpath . $sessid . "/";

//open log
$logpath = $usrpath . $sessid . "/" . $config->{'files'}->{'log'};
$log = fopen($logpath, 'a');
fwrite($log,",DEBUG,Begin scenario PHP script.\n");
fflush($log);

//write scenario parameter file
$fh = fopen($sesspath . $config->{'files'}->{'JSON'}->{'scenario'}, 'w');
fwrite($fh, $_POST["json"]);
fclose($fh);

//write model comments
$fh = fopen($sesspath . $config->{'files'}->{'comments'}, 'w');
fwrite($fh, $model->{'comments'});
fclose($fh);

//save initial run parameter file
move_uploaded_file($_FILES["init"]["tmp_name"],$sesspath . $config->{'files'}->{'JSON'}->{'init'});

//execute model
exec("nohup python " . $recpath. $config->{'paths'}->{'relative'}->{'core'} . $config->{'files'}->{'scenario'}
. " " . $sesspath . $config->{'files'}->{'JSON'}->{'scenario'}
. " " . $sesspath . $config->{'files'}->{'JSON'}->{'init'}
. " " . $sesspath . $config->{'paths'}->{'relative'}->{'predictors'}
. " " . "1> " . $sesspath . "std.log"
. " " . "2> " . $sesspath . "err.log &");

echo $sessid;

//symlink Flickr.csv and aoi_parms.csv
$initial = json_decode(file_get_contents($sesspath . $config->{'files'}->{'JSON'}->{'init'}));
exec("ln -s " . $usrpath . $initial->{'sessid'} . "/" . $config->{'files'}->{'flickr'} .
     " " . $sesspath . $config->{'files'}->{'flickr'});
exec("ln -s " . $usrpath . $initial->{'sessid'} . "/" . $config->{'files'}->{'params'} .
     " " . $sesspath . $config->{'files'}->{'params'});


//close log
fwrite($log,",DEBUG,End scenario PHP script.\n");
fclose($log);
?>
