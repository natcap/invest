from pathlib import Path
import time

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 1. Configure Chrome options to point to your Electron binary
chrome_options = Options()
for p in Path('.').glob(
    'dist/mac*/*.app/Contents/MacOS/InVEST*'):
    print(p)
    chrome_options.binary_location = str(p)

# 2. Connect to the ChromeDriver server running on port 9515
driver = webdriver.Remote(
    command_executor='http://localhost:9515',
    options=chrome_options
)

# Mocking variables from your environment setup
SCREENSHOT_PREFIX = "screenshot_"
TEST_PLUGIN_GIT_URL = "https://github.com/natcap/invest-demo-plugin.git"
TYPE_DELAY = 0.05  # Seconds (Puppeteer uses milliseconds)
TMP_DIR = "/tmp/workspace"

# Helper function to replicate Puppeteer's character typing delay
def send_keys_with_delay(element, text, delay):
    for char in text:
        element.send_keys(char)
        time.sleep(delay)

def test_install_and_run_plugin():

    # 1. Wait for the browser window to load and target 'index.html'
    # (Replaces BROWSER.waitForTarget)
    main_window_handle = None
    timeout = 30
    start_time = time.time()

    while time.time() - start_time < timeout:
        for handle in driver.window_handles:
            driver.switch_to.window(handle)
            if driver.current_url.endswith('index.html'):
                main_window_handle = handle
                break
        if main_window_handle:
            break
        time.sleep(1)

    if not main_window_handle:
        raise RuntimeError("Timed out waiting for index.html window target.")

    # 2. Take a screenshot
    driver.save_screenshot(f"{SCREENSHOT_PREFIX}1-page-load.png")
    time.sleep(2)

    modal_close = driver.find_element(
        By.XPATH, "//button[@aria-label='Close modal']")
    modal_close.click()
    time.sleep(2)

    modal_close = driver.find_element(
        By.XPATH, "//button[@aria-label='Close modal']"
    )
    modal_close.click()
    time.sleep(2)

    dropdown_button = driver.find_element(
        By.XPATH, "//button[@aria-label='menu']"
    )
    dropdown_button.click()
    time.sleep(2)

    plugins_modal_button = driver.find_element(
        By.XPATH, "//button[text()='Manage Plugins']"
    )
    plugins_modal_button.click()
    time.sleep(2)

    # 6. Type Git URL into Input Field
    url_input_field = driver.find_element(
        By.XPATH, "//input[@placeholder='https://github.com/owner/repo.git']"
    )
    send_keys_with_delay(url_input_field, TEST_PLUGIN_GIT_URL, TYPE_DELAY)

    # 7. Acknowledge and Submit
    user_acknowledgment_checkbox = driver.find_element(By.ID, "user-acknowledgment-checkbox")
    user_acknowledgment_checkbox.click()

    submit_button = driver.find_element(
        By.XPATH, "//button[text()='Add']"
    )
    submit_button.click()

    # wait for plugin to successfully install
    success_message = WebDriverWait(driver, 300).until(
        EC.presence_of_element_located((
            By.XPATH, "//*[text()='Successfully installed plugin']"
        ))
    )
    modal_close_button = driver.find_element(
        By.XPATH, "//button[@aria-label='Close modal']"
    )
    modal_close_button.click()

    plugin_button = driver.find_element(By.NAME, "Demo Plugin")
    plugin_button.click()

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((
            By.XPATH,
            "//div[contains(text(), 'Starting up model...')]"
        ))
    )

    # 10. Fill out Args Form
    args_form = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "args-form"))
    )

    # Note the use of dot (.) in Xpath to keep searches localized within args_form
    workspace_input = args_form.find_element(By.NAME, 'workspace_dir')
    send_keys_with_delay(workspace_input, TMP_DIR, TYPE_DELAY)

    raster_input = args_form.find_element(By.NAME, 'raster_path')
    send_keys_with_delay(
        raster_input,
        str(Path(__file__).resolve().parent / 'dem.tif'),
        TYPE_DELAY)

    number_input = args_form.find_element(By.NAME, 'factor')
    send_keys_with_delay(number_input, '2', TYPE_DELAY)

    # 11. Run Model
    run_button = driver.find_element(By.NAME, 'Run')
    run_button.click()

    # 12. Verify completion
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#invest-tab-tab-log.active"))
    )
    WebDriverWait(driver, 120).until(
        EC.presence_of_element_located((By.XPATH, "//div[contains(., 'Model Complete')]"))
    )

    driver.quit()
