-- This script is derived from the create-dmg utility
-- https://github.com/create-dmg/create-dmg
on run (volumeName)
    tell application "Finder"
        tell disk (volumeName as string)
            open

            -- location and size of DMG window when opened
            set theXOrigin to 100
            set theYOrigin to 100
            set theWidth to 900
            set theHeight to 660

            set theBottomRightX to (theXOrigin + theWidth)
            set theBottomRightY to (theYOrigin + theHeight)
            set dsStore to "\"" & "/Volumes/" & volumeName & "/" & ".DS_STORE\""

            tell container window
                set current view to icon view
                set toolbar visible to false
                set statusbar visible to false
                set the bounds to {theXOrigin, theYOrigin, theBottomRightX, theBottomRightY}
                set statusbar visible to false
                set position of every item to {theBottomRightX + 100, 100}
            end tell

            set opts to the icon view options of container window
            tell opts
                set icon size to 100
                set text size to 12
                set arrangement to not arranged
            end tell
            set background picture of opts to file ".background:background.png"

            -- Positioning
            set position of item "InVEST.app" to {220, 290}

            -- Applications link
            set position of item "Applications" to {670, 290}
            close
            open
            -- Force saving of the size
            delay 1

            tell container window
                set statusbar visible to false
                set the bounds to {theXOrigin, theYOrigin, theBottomRightX - 10, theBottomRightY - 10}
            end tell
        end tell

        delay 1

        tell disk (volumeName as string)
            tell container window
                set statusbar visible to false
                set the bounds to {theXOrigin, theYOrigin, theBottomRightX, theBottomRightY}
            end tell
        end tell

        --give the finder some time to write the .DS_Store file
        delay 3

        set waitTime to 0
        set ejectMe to false
        repeat while ejectMe is false
            delay 1
            set waitTime to waitTime + 1
            
            if (do shell script "[ -f " & dsStore & " ]; echo $?") = "0" then set ejectMe to true
        end repeat
        log "waited " & waitTime & " seconds for .DS_STORE to be created."
    end tell
end run
