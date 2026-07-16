from pathlib import Path
import platform
import shutil
import tempfile
import time
import unittest

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

CHROMEDRIVER_PORT = 9515
TEST_PLUGIN_GIT_URL = "https://github.com/natcap/invest-demo-plugin.git"

def send_keys_with_delay(element, text, delay):
    for char in text:
        element.send_keys(char)
        time.sleep(delay)

class PluginTests(unittest.TestCase):
    """Tests for the workbench plugin interface"""

    def setUp(self):
        self.workspace_dir = tempfile.mkdtemp()

        if platform.system() == 'Darwin':
            binary_glob = 'dist/mac*/*.app/Contents/MacOS/InVEST*'
        else:
            binary_glob = 'dist/win-unpacked/InVEST*.exe'
        binaries = list(Path(__file__).parent.parent.parent.glob(binary_glob))
        if len(binaries) > 1:
            raise ValueError('More than one binary found')

        # connect to the chromedriver server
        options = Options()
        options.binary_location = str(binaries[0].resolve())
        self.driver = webdriver.Remote(
            command_executor=f'http://localhost:{CHROMEDRIVER_PORT}',
            options=options)

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)

    def click(self, strategy, locator, timeout=5):
        element = WebDriverWait(self.driver, timeout).until(
            EC.element_to_be_clickable((strategy, locator)))
        element.click()

    def type(self, strategy, locator, text, typing_delay=0.05):
        element = self.driver.find_element(strategy, locator)
        send_keys_with_delay(element, text, typing_delay)

    def wait_for_main_window(self):
        """Cycle through open windows until the main window is found."""
        n_retries = 0
        while n_retries < 200:
            for handle in self.driver.window_handles:
                self.driver.switch_to.window(handle)
                if self.driver.current_url.endswith('index.html'):
                    return
                time.sleep(1)
                n_retries += 1
        raise RuntimeError(
            'Timed out waiting for index.html window target')

    def test_install_and_run_plugin(self):
        """Install and run the demo plugin."""

        self.driver.save_screenshot('screenshot1-page-load.png')
        self.wait_for_main_window()

        # close the "recent updates" and "download sample data" modals
        self.click(By.XPATH, "//button[@aria-label='Close modal']")
        self.click(By.XPATH, "//button[@aria-label='Close modal']")

        # click on the hamburger menu, then "Manage Plugins"
        self.click(By.XPATH, "//button[@aria-label='menu']")
        self.click(By.XPATH, "//button[text()='Manage Plugins']")

        # enter the plugin URL
        self.type(
            By.XPATH,
            "//input[@placeholder='https://github.com/owner/repo.git']",
            TEST_PLUGIN_GIT_URL)
        self.type(By.ID, "branch", 'update-invest')

        # check the acknowledgement and click the "Add" button
        self.click(By.ID, "user-acknowledgment-checkbox")
        self.click(By.XPATH, "//button[text()='Add']")

        # wait for plugin to successfully install, then
        # close the "Manage Plugins" modal
        WebDriverWait(self.driver, 300).until(
            EC.presence_of_element_located((
                By.XPATH, "//*[text()='Successfully installed plugin']"
            ))
        )
        self.click(By.XPATH, "//button[@aria-label='Close modal']")

        # launch the plugin
        self.click(By.NAME, "Demo Plugin")
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((
                By.XPATH,
                "//div[contains(text(), 'Starting up model...')]")))

        # enter input data into the form
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "args-form")))
        self.type(By.NAME, 'workspace_dir', self.workspace_dir)
        raster_path = str(Path(__file__).resolve().parent / 'dem.tif')
        self.type(By.NAME, 'raster_path', raster_path)
        self.type(By.NAME, 'factor', '2')

        # run the model and wait for it to complete
        self.click(By.NAME, 'Run')
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "#invest-tab-tab-log.active")))
        WebDriverWait(self.driver, 120).until(
            EC.presence_of_element_located(
                (By.XPATH, "//div[contains(., 'Model Complete')]")))

        self.driver.quit()
