Icon (.icns) files
------------------

Applications needed:
  * Image2Icon (http://www.img2icnsapp.com/)
  * ICNSmini (https://itunes.apple.com/us/app/icnsmini-shrink-png-icns-iconsets/id1035260885?mt=12)

I've gotten good results from the following workflow:
    * Use the .ico file we're using for the InVEST Windows Installer
    * Add the .ico file to the mac application image2icon and export a .icns file.
    * Take the icns file exported from image2icon and add it to ICNSmini.  This will
      compress the ICNS file.
    * Use the ICNS file output from ICNSmini in the mac installer file.
