out_file=HISTORY.rst
if [ -e $out_file ]
then
    dos2unix -1252 $out_file
fi
REPO=src/invest-natcap.default

hg revert --all -R $REPO
find $REPO -name "*.orig" | xargs rm

for file in `ls src/invest-natcap.default/docs/release_notes/Updates_InVEST_{2,3}* | sort -r`
do
    version=$(echo $file | grep -o [0-9].[0-9].[0-9].txt | sed 's/_/ /g' | sed 's/.txt//g')
    tagdate=$(hg log -R $REPO -r "$version" --template="{date(date, '%Y-%m-%d')}")
    echo $version \($tagdate\) >> $out_file
    echo "------------------">> $out_file
    dos2unix -1252 $file  # assume the file is written in windows codepage 1252.
    cat $file >> $out_file
    echo  >> $out_file
    echo  >> $out_file
done
