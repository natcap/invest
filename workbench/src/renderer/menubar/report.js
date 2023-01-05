import {
  handleClickExternalURL,
  handleClickFindLogfiles
} from './handlers';

document.querySelector('button').addEventListener('click', handleClickFindLogfiles);
document.querySelectorAll('a').forEach(
  (element) => {
    element.addEventListener('click', handleClickExternalURL);
  }
);
