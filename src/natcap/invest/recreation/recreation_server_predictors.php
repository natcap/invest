<?php

//http://www.nusphere.com/kb/phpmanual/function.ini-get.htm
function return_bytes($val) {
    $val = trim($val);
    $last = strtolower($val{strlen($val)-1});
    switch($last) {
        // The 'G' modifier is available since PHP 5.1.0
        case 'g':
            $val *= 1024;
        case 'm':
            $val *= 1024;
        case 'k':
            $val *= 1024;
    }

    return $val;
}


//read configuration
$json = file_get_contents("./recreation_server_config.json");
$config = json_decode($json);

//set paths
$usrpath = $config->{'paths'}->{'absolute'}->{'userroot'} . $config->{'paths'}->{'relative'}->{'data'};
$predpath = $config->{'paths'}->{'relative'}->{'predictors'};

//get session id
$sessid = $_POST["sessid"];
$sesspath = $usrpath . $sessid . "/";

//open log
$logpath = $sesspath . $config->{'files'}->{'log'};
$log = fopen($logpath, 'a');
fwrite($log,",DEBUG,Begin predictors PHP script.\n");
fflush($log);

$size = (int) $_SERVER['CONTENT_LENGTH'];
$max_size = return_bytes(ini_get('post_max_size'));

fwrite($log,",INFO,You have uploaded " . sizeof($_FILES) . " files.\n");
fflush($log);

fwrite($log,",INFO,Your upload size is " . $size . ".\n");
fflush($log);

fwrite($log,",INFO,The max upload size is " . $max_size . ".\n");
fflush($log);

if ($size>$max_size)
{
  fwrite($log,",ERROR,The max upload has been exceeded.\n");
  fflush($log);
}
else
{
  fwrite($log,",DEBUG,The upload size is acceptable.\n");
  fflush($log);

  $zip = new ZipArchive;
  $zip->open($_FILES["zip_file"]["tmp_name"], ZipArchive::CHECKCONS);
  $zip->extractTo($sesspath . $predpath );
  $zip->close();

}

echo $sessid;

//close log
fwrite($log,",DEBUG,End predictors PHP script.\n");
fclose($log);
?>
