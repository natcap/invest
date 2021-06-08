import fs from 'fs';
import path from 'path';

import yauzl from 'yauzl';

import { getLogger } from '../logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

export default function extractZipInplace(zipFilePath) {
  return new Promise((resolve, reject) => {
    try {
      const extractToDir = path.dirname(zipFilePath);
      logger.info(`extracting ${zipFilePath}`);
      const options = {
        lazyEntries: true,
        autoClose: true,
      };
      yauzl.open(zipFilePath, options, (err, zipfile) => {
        if (err) throw err;
        zipfile.on('entry', (entry) => {
          const writePath = path.join(extractToDir, entry.fileName);
          // if entry is a directory
          if (/\/$/.test(entry.fileName)) {
            fs.mkdir(writePath, (err) => {
              if (err) {
                if (err.code === 'EEXIST') { } else throw err;
              }
              zipfile.readEntry();
            });
          } else {
            zipfile.openReadStream(entry, (error, readStream) => {
              if (error) throw error;
              readStream.on('end', () => {
                zipfile.readEntry();
              });
              // Sometimes an entry will be in a dir, where the
              // dir itself was *not* an entry, therefore we still need
              // to create the dir here.
              fs.mkdir(path.dirname(writePath), (err) => {
                if (err) {
                  if (err.code === 'EEXIST') { } else throw err;
                }
                const writable = fs.createWriteStream(writePath);
                readStream.pipe(writable);
              });
              // TODO: an ENOENT was being raised here
            });
          }
        });
        zipfile.on('close', () => {
          resolve(true);
        });
        zipfile.readEntry();
      });
    } catch (error) {
      // TODO, even though an ENOENT was raised - prob from fs.createWriteStream
      // this catch and reject never ran. wtf.
      reject(error);
    }
  });
}
