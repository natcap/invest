import { useEffect, useRef, useState, type ChangeEvent, type RefObject } from 'react';

import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import { useTranslation } from 'react-i18next';
import { MdClose } from 'react-icons/md';

import {
  transformDHALSearchResult,
  type DataHubSearchResult,
  type DHALSearchResult
} from './models';
import {
  DHALSearchParams,
  type DataHubSearchQuery
} from './DataHubSearchParams/models';
import {
  DataHubSearchIntroContent,
  DataHubSearchIntroFooter,
  DataHubSearchLoadingContent,
  DataHubSearchResultsContent,
  DataHubSearchResultsFooter,
  DataHubSearchSearchingContent
} from './DataHubSearchModalViews';
import DataHubSearchParams from './DataHubSearchParams';

// @ts-ignore
const { logger } = window.Workbench;

interface DataHubSearchModalProps {
  show: boolean,
  closeModal: () => {},
  query: DataHubSearchQuery,
  aoiInputName: string,
  argsValidationComplete: boolean,
  aoiIsValid: boolean,
  updateSearchExtent: () => {},
  searchExtentUpdateComplete: boolean,
  selectSearchResult: (url: string, collections: string[]) => {},
  requestFocusOnAoiInput: () => {},
}

const DHAL_BASE_URL = 'https://data.naturalcapitalalliance.stanford.edu/dhal/search_dataset/';

enum searchStep {
  LOADING,
  INTRO,
  SEARCHING,
  RESULTS,
}

export const searchModalContentSummaryCssClass = 'search-modal-content-summary';
export const searchModalAutoFocusId = 'search-modal-auto-focus';
const autoFocusSelector = `#${searchModalAutoFocusId}`;

export default function DataHubSearchModal(props: DataHubSearchModalProps) {
  const {
    show,
    closeModal,
    query,
    aoiInputName,
    argsValidationComplete,
    aoiIsValid,
    updateSearchExtent,
    searchExtentUpdateComplete,
    selectSearchResult,
    requestFocusOnAoiInput,
  } = props;

  const { t } = useTranslation();

  const getLoadingOrIntroStep = (): searchStep => {
    if (
      query.datatype === 'raster'
      && (!argsValidationComplete || (aoiIsValid && !searchExtentUpdateComplete))
    ) {
      // Disallow searching for a raster while waiting for AOI validation or
      // for a valid extent.
      return searchStep.LOADING;
    } else {
      // Allow search or render "Invalid AOI" error as appropriate.
      return searchStep.INTRO;
    }
  }

  const [step, setStep] = useState<searchStep>(getLoadingOrIntroStep());
  const [numSearchResults, setNumSearchResults] = useState<number>(0);
  const [searchResults, setSearchResults] = useState<DataHubSearchResult[]>([]);
  const [allExpanded, setAllExpanded] = useState<boolean>(false);
  const [cardsExpanded, setCardsExpanded] = useState<Map<string, boolean>>(new Map());
  const [searchError, setSearchError] = useState<boolean>(false);

  const autoFocusRef: RefObject<any> = useRef(null);

  const searchAllowed: boolean = aoiIsValid || query.datatype === 'csv';

  useEffect(() => {
    if (query.datatype === 'raster' && argsValidationComplete) {
      updateSearchExtent();
    }
  }, [argsValidationComplete]);

  useEffect(() => {
    setStep(getLoadingOrIntroStep());
  }, [argsValidationComplete, aoiIsValid, searchExtentUpdateComplete]);

  useEffect(() => {
    // This effect supports screen reader navigation by auto-focusing the
    // element (typically a heading) that provides a summary of new content,
    // whenever modal content changes. If a designated content summary element
    // does not exist, the modal itself receives focus.
    if (show) {
      const contentSummaryElement: HTMLElement | null = autoFocusRef.current?.dialog?.querySelector(autoFocusSelector);
      if (contentSummaryElement) {
        contentSummaryElement.focus();
      } else {
        autoFocusRef.current?.dialog?.focus();
      }
    }
  }, [show, step]);

  const search = async () => {
    setStep(searchStep.SEARCHING);
    setSearchError(false);

    const params = new DHALSearchParams(query);
    const requestUrl = `${DHAL_BASE_URL}?${params}`;

    try {
      const response = await fetch(requestUrl);
      const responseBody = await response.json();
      if (!response.ok) {
        throw new Error(
          `HTTP request failed.
          Request URL: ${requestUrl}
          HTTP error ${response.status} (${response.statusText}): ${responseBody.error_message}`
        );
      } else {
        const {count, datasets} = responseBody;
        setNumSearchResults(count);
        setSearchResults(datasets.map(
          (d: DHALSearchResult) => transformDHALSearchResult(d))
        );
        collapseAllCards();
        logger.info(
          `HTTP request succeeded.
          Request URL: ${requestUrl}`
        );
      }
    } catch (error) {
      setSearchError(true);
      logger.error((error as Error).message);
    } finally {
      setStep(searchStep.RESULTS);
    }
  };

  const goToAoiInput = () => {
    requestFocusOnAoiInput();
    close();
  };

  const toggleExpandCard = (id: string) => {
    const prevState = cardsExpanded.get(id);
    setCardsExpanded(new Map([...cardsExpanded, [id, !prevState]]));
  };

  const toggleExpandAll = (event: ChangeEvent) => {
    const checkbox = event.target as HTMLInputElement;
    setAllExpanded(checkbox.checked);
  };

  const expandAllCards = () => {
    setCardsExpanded(new Map(searchResults.map(({id}) => [id, true])));
  };

  const collapseAllCards = () => {
    setCardsExpanded(new Map(searchResults.map(({id}) => [id, false])));
  };

  const selectDataset = (url: string, collections: string[]) => {
    selectSearchResult(url, collections);
    close();
  };

  const close = () => {
    // Reset modal to loading or intro step each time it closes.
    // @TODO: Consider preserving step number to prevent repeated user
    // interactions and/or repeated Data Hub queries, resetting step only
    // if/when query params have changed.
    setStep(getLoadingOrIntroStep());
    closeModal();
  };

  useEffect(() => {
    allExpanded ? expandAllCards() : collapseAllCards();
  }, [allExpanded]);

  return (
    <Modal
      show={show}
      onHide={close}
      scrollable
      contentClassName="search-modal"
      ref={autoFocusRef}
    >
      <Modal.Header>
        <Modal.Title as="h1" className="h4">{t('Search the Data Hub')}</Modal.Title>
        <Button
          variant="secondary-outline"
          onClick={close}
          aria-label={t('Close modal')}
        >
          <MdClose />
        </Button>
      </Modal.Header>
      <Modal.Body>
        {
          step === searchStep.LOADING &&
          <DataHubSearchLoadingContent />
        }
        {
          step !== searchStep.LOADING && searchAllowed &&
          <>
            <DataHubSearchParams
              tags={query.tags}
              datatype={query.datatype}
              extent={query.extent}
              collections={query.collections}
            />
          </>
        }
        {
          step === searchStep.INTRO &&
          <DataHubSearchIntroContent
            aoiInputName={aoiInputName}
            searchAllowed={searchAllowed}
          />
        }
        {
          step === searchStep.SEARCHING &&
          <DataHubSearchSearchingContent />
        }
        {
          step === searchStep.RESULTS &&
          <DataHubSearchResultsContent
            searchError={searchError}
            numSearchResults={numSearchResults}
            searchResults={searchResults}
            toggleExpandCard={toggleExpandCard}
            toggleExpandAll={toggleExpandAll}
            cardsExpanded={cardsExpanded}
            selectDataset={selectDataset}
          />
        }
      </Modal.Body>
      {
        step === searchStep.INTRO &&
        <Modal.Footer>
          <DataHubSearchIntroFooter
            searchAllowed={searchAllowed}
            aoiInputName={aoiInputName}
            search={search}
            close={close}
            goToAoiInput={goToAoiInput}
          />
        </Modal.Footer>
      }
      {
        step === searchStep.RESULTS && searchError &&
        <Modal.Footer>
          <DataHubSearchResultsFooter
            searchError={searchError}
            search={search}
          />
        </Modal.Footer>
      }
    </Modal>
  );
}
