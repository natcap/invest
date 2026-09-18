import { useEffect, useRef, useState, type ChangeEvent, type RefObject } from 'react';

import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import { useTranslation } from 'react-i18next';
import { MdClose } from 'react-icons/md';

import {
  transformDHALSearchResult,
  type DataHubSearchResult,
  type DHALDataset
} from './models';
import {
  DHALSearchParams,
  type DataHubSearchQuery
} from './DataHubSearchParams/models';
import {
  DataHubSearchIntroBody,
  DataHubSearchIntroFooter,
  DataHubSearchResultsBody,
  DataHubSearchResultsFooter,
  DataHubSearchSearchingBody
} from './DataHubSearchModalViews';
import DataHubSearchParams from './DataHubSearchParams';

// @ts-ignore
const { logger } = window.Workbench;

interface DataHubSearchModalProps {
  show: boolean,
  closeModal: () => {},
  query: DataHubSearchQuery,
  aoiInputName: string,
  aoiIsValid: boolean,
  selectSearchResult: (url: string, collections: string[]) => {},
  requestFocusOnAoiInput: () => {},
}

const DHAL_BASE_URL = 'https://data.naturalcapitalalliance.stanford.edu/dhal/search_dataset/';

const INTRO_STEP = 0;
const SEARCHING_STEP = 1;
const RESULTS_STEP = 2;

export const searchModalContentHeadingCssClass = 'search-modal-content-heading';
export const searchModalAutoFocusId = 'search-modal-auto-focus';
const autoFocusSelector = `#${searchModalAutoFocusId}`;

export default function DataHubSearchModal(props: DataHubSearchModalProps) {
  const {
    show,
    closeModal,
    query,
    aoiInputName,
    aoiIsValid,
    selectSearchResult,
    requestFocusOnAoiInput,
  } = props;

  const { t } = useTranslation();

  const [step, setStep] = useState<number>(INTRO_STEP);
  const [numSearchResults, setNumSearchResults] = useState<number>(0);
  const [searchResults, setSearchResults] = useState<DataHubSearchResult[]>([]);
  const [allExpanded, setAllExpanded] = useState<boolean>(false);
  const [cardsExpanded, setCardsExpanded] = useState<Map<string, boolean>>(new Map());
  const [searchError, setSearchError] = useState<boolean>(false);

  const autoFocusRef: RefObject<any> = useRef(null);

  useEffect(() => {
    // This effect supports screen reader navigation by auto-focusing the
    // element (typically a heading) that provides a summary of new content,
    // whenever modal content changes. If a designated content summary element
    // does not exist, the modal itself receives focus.
    const contentSummaryElement: HTMLElement | null = autoFocusRef.current?.dialog?.querySelector(autoFocusSelector);
    if (contentSummaryElement) {
      contentSummaryElement.focus();
    } else {
      autoFocusRef.current?.dialog?.focus();
    }
  }, [step]);

  const search = async () => {
    setStep(SEARCHING_STEP);
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
        setSearchResults(datasets.map((d: DHALDataset) => transformDHALSearchResult(d)));
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
      setStep(RESULTS_STEP);
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
    // Setting step to 0 "resets" modal state each time it closes.
    // @TODO: ¿consider preserving step number to prevent repeated user interactions,
    // perhaps resetting step number only if/when query params have changed?
    setStep(0);
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
          aoiIsValid &&
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
          step === INTRO_STEP &&
          <DataHubSearchIntroBody
            aoiInputName={aoiInputName}
            aoiIsValid={aoiIsValid}
          />
        }
        {
          step === SEARCHING_STEP &&
          <DataHubSearchSearchingBody />
        }
        {
          step === RESULTS_STEP &&
          <DataHubSearchResultsBody
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
        step === INTRO_STEP &&
        <Modal.Footer>
          <DataHubSearchIntroFooter
            aoiIsValid={aoiIsValid}
            aoiInputName={aoiInputName}
            search={search}
            close={close}
            goToAoiInput={goToAoiInput}
          />
        </Modal.Footer>
      }
      {
        step === RESULTS_STEP && searchError &&
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
