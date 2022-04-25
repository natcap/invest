import fs from 'fs';
import path from 'path';

import yauzl from 'yauzl';

import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

/** Extract a zip archive to a directory with the same basename.
 *
 * Handle an arbitrary depth of files and folders within the archive,
 * mirroring that structure in the output directory.
 *
 * @param {string} zipFilePath - path to the local zipfile.
 * @returns { Promise } resolves true when done extracting all contents.
 */
export default function extractZipInplace(zipFilePath) {
  return new Promise((resolve) => {
    const extractToDir = path.dirname(zipFilePath);
    logger.info(`extracting ${zipFilePath}`);
    // lazyEntries allows explicit calls of readEntry,
    // which we need to because we need to setup dirs as we go.
    const options = {
      lazyEntries: true,
    };
    yauzl.open(zipFilePath, options, (error, zipfile) => {
      if (error) throw error;
      zipfile.on('entry', (entry) => {
        const writePath = path.join(extractToDir, entry.fileName);
        // if entry is a directory
        if (/\/$/.test(entry.fileName)) {
          fs.mkdir(writePath, (e) => {
            if (e) {
              if (e.code !== 'EEXIST') { throw e; }
            }
            zipfile.readEntry();
          });
        } else {
          zipfile.openReadStream(entry, (err, readStream) => {
            if (err) throw err;
            readStream.on('end', () => {
              zipfile.readEntry();
            });
            // Sometimes an entry will be in a dir, where the
            // dir itself was *not* an entry, therefore we still need
            // to create the dir (and possibly all dirs in the chain) here.
            fs.mkdir(path.dirname(writePath), { recursive: true }, (e) => {
              if (e) {
                if (e.code !== 'EEXIST') { throw e; }
              }
              const writable = fs.createWriteStream(writePath);
              readStream.pipe(writable);
            });
          });
        }
      });
      zipfile.on('close', () => {
        resolve(true);
      });
      zipfile.readEntry();
    });
  });
}
