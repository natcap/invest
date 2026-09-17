import type { ChangeEvent } from 'react';

import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import Spinner from 'react-bootstrap/esm/Spinner';
import { useTranslation } from 'react-i18next';
import { TbMapSearch, TbZoomCancel, TbZoomExclamation } from 'react-icons/tb';

import DataHubSearchParams from '../DataHubSearchParams';
import type { DataHubSearchQuery } from '../DataHubSearchParams/models';
import type { DataHubSearchResult } from '../models';
import DataHubSearchResultCard from '../DataHubSearchResultCard';

import { openLinkInBrowser } from '../../../utils';
import { handleClickFindLogfiles } from '../../../menubar/handlers';

interface IntroBodyProps {
  aoiInputName: string,
  aoiIsValid: boolean,
  query: DataHubSearchQuery,
}

export function DataHubSearchIntroBody(props: IntroBodyProps) {
  const { aoiIsValid, aoiInputName, query } = props;
  const { t } = useTranslation();

  return (
      <>
        <p>
          {t(`Search the Natural Capital Alliance Data Hub for datasets you can
            use in InVEST without having to download them first.`)}
        </p>
        {/* @TODO: add loading spinner if/when awaiting AOI validation status and/or AOI extent. */}
      {
        aoiIsValid
        ? <DataHubSearchParams
            tags={query.tags}
            datatype={query.datatype}
            extent={query.extent}
            collections={query.collections}
          />
        : <>
            <div className="search-error">
              <TbZoomCancel aria-label={t('Error')} className="error-icon" />
              <span>
                {t(`Before searching, you must specify a valid path for the following input:`)}
                <strong className="aoi-input-name">{aoiInputName}</strong>
              </span>
            </div>
          </>
        }
      </>
  );
}

export function DataHubSearchSearchingBody() {
  return (
    <>
      <Spinner animation="border" role="status" className="search-spinner"></Spinner>
    </>
  );
}

interface ResultsBodyProps {
  query: DataHubSearchQuery,
  searchError: boolean,
  numSearchResults: number,
  searchResults: DataHubSearchResult[],
  toggleExpandCard: (id: string) => void,
  toggleExpandAll: (event: ChangeEvent) => void,
  cardsExpanded: Map<string, boolean>,
  selectDataset: (url: string, collections: string[]) => void,
}

export function DataHubSearchResultsBody(props: ResultsBodyProps) {
  const {
    query, searchError, numSearchResults, searchResults,
    toggleExpandCard, toggleExpandAll, cardsExpanded, selectDataset } = props;
  const { t } = useTranslation();

  return (
    <>
      <DataHubSearchParams
        tags={query.tags}
        datatype={query.datatype}
        extent={query.extent}
        collections={query.collections}
      />
      {
        searchError
        ? <>
            <div className="search-error">
              <TbZoomExclamation aria-label={t('Error')} className="error-icon" />
              <span>
                {t(`An error occurred. Please check your internet connection, then try again.
                  If the problem persists, consider reporting it on the NatCap Community Forum.`)}
              </span>
            </div>
            <a
              href="https://community.naturalcapitalalliance.org/"
              className="d-flex justify-content-center"
              onClick={openLinkInBrowser}
            >
              {t('Natural Capital Alliance Community Forum (opens in web browser)')}
            </a>
          </>
        : (
            numSearchResults
            ? <>
                <div className="search-results-header">
                  <p>{numSearchResults == 1 ? t('1 result found.') : t(`${numSearchResults} results found.`)}</p>
                  <Form.Check
                    type="switch" // type="switch" controls styling
                    role="switch" // role="switch" communicates correct semantics to assistive tech
                    id="expand-all"
                    label={t('Expand All')}
                    onChange={toggleExpandAll}
                  />
                </div>
                <div className="search-results">
                  {searchResults.map((result =>
                    <DataHubSearchResultCard
                      key={result.id}
                      datasetDetails={result}
                      expanded={cardsExpanded.get(result.id) ? true : false}
                      onToggleExpanded={() => toggleExpandCard(result.id)}
                      onSelect={selectDataset}
                    />
                  ))}
                </div>
              </>
            : <>
                <p>{t('No results found.')}</p>
                <TbMapSearch className="no-results-icon" aria-hidden="true" />
                <p>
                  {t(`We are actively working on adding more datasets to the Data Hub
                  to meet the needs of InVEST users. Please check back later as the
                  collection grows!`)}
                </p>
                <p>
                  {t(`In the meantime, if you'd like to explore the Data Hub on your
                    own, you can visit it on the web:`)}
                  <a
                    href="https://data.naturalcapitalalliance.stanford.edu/"
                    className="d-flex"
                    onClick={openLinkInBrowser}
                  >
                    {t(`Natural Capital Alliance Data Hub (opens in web browser)`)}
                  </a>
                </p>
              </>
          )
      }
    </>
  );
}

export function DataHubSearchIntroFooter(
  props: {
    aoiIsValid: boolean,
    aoiInputName: string,
    search: () => void,
    close: () => void,
    goToAoiInput: () => void,
  }
) {
  const { aoiIsValid, aoiInputName, search, close, goToAoiInput } = props;
  const { t } = useTranslation();

  return (
      aoiIsValid
      ? <Button onClick={search}>{t('Search')}</Button>
      : <>
          <Button variant="outline-primary" onClick={close}>{t('OK')}</Button>
          <Button onClick={goToAoiInput}>{t(`Go to ${aoiInputName} form field`)}</Button>
        </>
  );
}

export function DataHubSearchResultsFooter(
  props: {searchError: boolean, search: () => void}
) {
  const { searchError, search } = props;
  const { t } = useTranslation();

  return (
      searchError &&
      <>
          <Button variant="outline-primary" onClick={handleClickFindLogfiles}>
            {t('Find my logs')}
          </Button>
        <Button onClick={search}>{t('Search again')}</Button>
      </>
  );
}
