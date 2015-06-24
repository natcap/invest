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
fwrite($log,",DEBUG,Begin recreation PHP script.\n");
fflush($log);

//write model parameters
$fh = fopen($sesspath . $config->{'files'}->{'JSON'}->{'init'}, 'w');
fwrite($fh, $_POST["json"]);
fclose($fh);

//stop if model includes restricted datasets
$json = json_decode($_POST["json"]);
if ($json->{'global_data'}=="True")
{
    if ($json->{'protected'}=="True"){
        fwrite($log,",ERROR,Sorry due to licensing restrictions protected areas and ocean use and ocean coverage are not currently available.");
        fflush($log);
        fclose($log);
        exit;
    }

    if ($json->{'ouoc'}=="True"){
        fwrite($log,",ERROR,Sorry due to licensing restrictions protected areas and ocean use and ocean coverage are not currently available.");
        fflush($log);
        fclose($log);
        exit;
    }    
}

//write model comments
$fh = fopen($sesspath . $config->{'files'}->{'comments'}, 'w');
fwrite($fh, $model->{'comments'});
fclose($fh);

if ($model->{'download'}){
mkdir($sesspath . $config->{'paths'}->{'relative'}->{'download'} , 0755);
}

//save uploaded area of interest
move_uploaded_file($_FILES["aoiSHP"]["tmp_name"],$sesspath . $config->{'files'}->{'aoi'}->{'shp'});
move_uploaded_file($_FILES["aoiSHX"]["tmp_name"],$sesspath . $config->{'files'}->{'aoi'}->{'shx'});
move_uploaded_file($_FILES["aoiDBF"]["tmp_name"],$sesspath . $config->{'files'}->{'aoi'}->{'dbf'});
move_uploaded_file($_FILES["aoiPRJ"]["tmp_name"],$sesspath . $config->{'files'}->{'aoi'}->{'prj'});

/*
//save uploaded area of interest
move_uploaded_file($_FILES["aoiSHP"]["tmp_name"],$sesspath . "tmp.shp");
move_uploaded_file($_FILES["aoiSHX"]["tmp_name"],$sesspath . "tmp.shx");
move_uploaded_file($_FILES["aoiDBF"]["tmp_name"],$sesspath . "tmp.dbf");
move_uploaded_file($_FILES["aoiPRJ"]["tmp_name"],$sesspath . "tmp.prj");

$command="ogr2ogr -t_srs EPSG:4326 " . $sesspath . $config->{'files'}->{'aoi'}->{'shp'} . " " . $sesspath . "tmp.shp" ;
fwrite($log,"\n" . "Executing command: " . $command);
fflush($log);
exec($command);
*/

//execute model
exec("touch ". $sesspath . $config->{'files'}->{'log'});
$command="nohup python " . $config->{'paths'}->{'absolute'}->{'repository'}
. $config->{'paths'}->{'relative'}->{'core'}
. $config->{'files'}->{'init'}
. " " . $sesspath . $config->{'files'}->{'JSON'}->{'init'}
. " " . $sesspath . "grid.shp"
. " " . $sesspath . $config->{'files'}->{'flickr'}
. " " . $sesspath . $config->{'paths'}->{'relative'}->{'predictors'}
. " " . "1> " . $sesspath . "std.log"
. " " . "2> " . $sesspath . "err.log &";
fwrite($log,",INFO,Executing recreation Python script.\n");
fflush($log);
fwrite($log,",DEBUG,Executing command: " . str_replace(".", "||", str_replace(",", "|", $command)) . ".\n");
fflush($log);
exec($command);

echo $sessid;

//close log
fwrite($log,",DEBUG,End recreation PHP script.\n");
fclose($log);
?>
