import { useTranslation } from 'react-i18next';

import type { DataHubSearchQuery } from './models';

export default function DataHubSearchParams(query: DataHubSearchQuery) {
  const { t } = useTranslation();

  return (
    <dl className="search-params">
      <dt>{t('Tags')}</dt>
      <dd>{query.tags.join(', ')}</dd>
      <dt>{t('Datatype')}</dt>
      <dd>{query.datatype}</dd>
      {
        query.extent.length > 0 &&
        <>
          <dt>{t('Extent')}</dt>
          <dd>{query.extent.join(', ')}</dd>
        </>
      }
      {
        query.collections.length > 0
        ? <>
            <dt>{t('Collections')}</dt>
            <dd>{query.collections.join(', ')}</dd>
          </>
        : <></>
      }
    </dl>
  );
}
