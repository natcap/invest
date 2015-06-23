# make soft links for each of the invest exe's into /usr/bin.

for exe in `find /usr/lib/invest-bin -executable -type f`
do
    ln -s /usr/bin/`basename $exe` $exe
done
