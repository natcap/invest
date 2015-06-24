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

$recpath = $config->{'paths'}->{'absolute'}->{'repository'};
$compath = $config->{'paths'}->{'relative'}->{'computation'};
$regpath = $recpath . $compath . $config->{'files'}->{'regression'};

//open log
$logpath = $sesspath . $config->{'files'}->{'log'};
$log = fopen($logpath, 'a');
fwrite($log,",DEBUG,Begin regression PHP script.\n");
fflush($log);

//build array of restricted columns
fwrite($log,",DEBUG,Restricted columns active.\n"); 
$excludecol=$config->{'postgis'}->{'table'}->{'restrict'};
foreach ($excludecol as $col => $value){
fwrite($log,",DEBUG,Restricted column: " . $value . ".\n");
}
fflush($log);

//execute regression
$command="nohup R CMD BATCH --vanilla '--args sessid=\"".$sessid."\"' ".$regpath." ".$sesspath.$config->{'files'}->{'Rlog'} . " &";
fwrite($log,",DEBUG,Executing command: " . str_replace(".", "||", str_replace(",", "|", $command)) . ".\n");
fflush($log);
exec($command);

echo $sessid;

//close log
fwrite($log,",DEBUG,End regression PHP script.\n");
fclose($log);
?>
