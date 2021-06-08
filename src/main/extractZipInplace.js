import fs from 'fs';
import path from 'path';

import yauzl from 'yauzl';

import { getLogger } from '../logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

export default function extractZipInplace(zipFilePath) {
  console.log('extracting')
  return new Promise((resolve, reject) => {
    try {
      const extractToDir = path.dirname(zipFilePath);
      logger.info(`extracting ${zipFilePath}`);
      const options = {
        lazyEntries: true,
        autoClose: true,
      };
      yauzl.open(zipFilePath, options, (err, zipfile) => {
        console.log('yauzl.open')
        if (err) throw err;
        zipfile.on('entry', (entry) => {
          console.log('zipfile.on entry')
          const writePath = path.join(extractToDir, entry.fileName);
          // if entry is a directory
          if (/\/$/.test(entry.fileName)) {
            console.log('entry is a dir')
            fs.mkdir(writePath, (err) => {
              console.log('fs.mkdir callback for directory')
              if (err) {
                if (err.code === 'EEXIST') { } else throw err;
              }
              zipfile.readEntry();
              console.log('zipfile.readEentry with dir')
            });
          } else {
            console.log('entry is a file')
            zipfile.openReadStream(entry, (error, readStream) => {
              console.log('openReadStream')
              if (error) throw error;
              readStream.on('end', () => {
                console.log('readStream on end')
                zipfile.readEntry();
                console.log('zipfile next readEntry')
              });
              // Sometimes an entry will be in a dir, where the
              // dir itself was *not* an entry, therefore we still need
              // to create the dir here.
              fs.mkdir(path.dirname(writePath), (err) => {
                console.log('fs.mkdir for file with a parent dir')
                if (err) {
                  if (err.code === 'EEXIST') { } else throw err;
                }
              });
              const writable = fs.createWriteStream(writePath);
              console.log('fs.createWriteStream')
              readStream.pipe(writable);
              console.log('reaedStream.pipe')
            });
          }
        });
        zipfile.on('close', () => {
          console.log('zipfile on close, resolving')
          resolve(true);
        });
        zipfile.readEntry();
        console.log('zipfile.readEntry')
      });
    } catch (error) {
      console.log('rejecting');
      reject(error);
    }
  });
}
