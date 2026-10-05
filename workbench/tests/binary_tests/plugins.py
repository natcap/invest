import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
import unittest

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

CHROMEDRIVER_PORT = 9515
TEST_PLUGIN_LOCAL_PATH = str(Path(__file__).parent / "test_plugin")


class PluginTests(unittest.TestCase):
    """Tests for the workbench plugin interface"""

    def setUp(self):
        self.workspace_dir = tempfile.mkdtemp()

        if platform.system() == 'Darwin':
            binary_glob = 'dist/mac*/*.app/Contents/MacOS/InVEST*'
            self.config_path = Path(
                '~/Library/Application Support/invest-workbench/config.json'
            ).expanduser()
        else:
            binary_glob = 'dist/win-unpacked/InVEST*.exe'
            self.config_path = os.path.expandvars(
                '%APPDATA%/invest-workbench/config.json')
        binaries = list(Path(__file__).parent.parent.parent.glob(binary_glob))
        if len(binaries) > 1:
            raise ValueError('More than one binary found')

        # connect to the chromedriver server
        options = Options()
        options.binary_location = str(binaries[0].resolve())
        # options.add_argument("--headless=new")      # Runs Chrome in headless mode
        # options.add_argument("--no-sandbox")         # Bypasses OS security model layer
        # options.add_argument("--disable-dev-shm-usage")  # Overcomes limited resource problems in Docker
        # options.add_argument("--disable-gpu")        # Temporary fix for certain hardware environments
        self.driver = webdriver.Remote(
            command_executor=f'http://localhost:{CHROMEDRIVER_PORT}',
            options=options)

    def tearDown(self):
        shutil.rmtree(self.workspace_dir)
        self.driver.quit()

    def click(self, strategy, locator, timeout=5):
        """Click on an element once it is clickable"""
        element = WebDriverWait(self.driver, timeout).until(
            EC.element_to_be_clickable((strategy, locator)))
        element.click()

    def type(self, strategy, locator, text, typing_delay=0.01):
        """Enter text into an input field, with delay to simulate typing."""
        element = self.driver.find_element(strategy, locator)
        for char in text:
            element.send_keys(char)
            time.sleep(typing_delay)

    def wait_for_main_window(self, max_retries=200):
        """Cycle through open windows until the main window is found."""
        n_retries = 0
        while n_retries < max_retries:
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

        print('starting test')
        try:
            # close the "recent updates" and "download sample data" modals
            self.click(By.XPATH, "//button[@aria-label='Close modal']")
            self.click(By.XPATH, "//button[@aria-label='Close modal']")
        except Exception:
            pass

        # click on the hamburger menu, then "Manage Plugins"
        self.click(By.XPATH, "//button[@aria-label='menu']")
        self.click(By.XPATH, "//button[text()='Manage Plugins']")

        # enter the plugin URL
        Select(
            self.driver.find_element(By.ID, "installFrom")
        ).select_by_visible_text("local path")
        self.type(
            By.XPATH,
            "//input[@placeholder='/Users/username/path/to/plugin/']",
            TEST_PLUGIN_LOCAL_PATH)

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

        # install the current dev branch into the plugin environment
        self.click(By.XPATH, "//button[@aria-label='menu']")
        self.click(By.XPATH, "//button[text()='Manage Plugins']")
        env_path = self.driver.find_element(
            By.ID, "test@0_0_0").get_attribute("value")
        subprocess.run([
            'micromamba', 'run', '--prefix', env_path,
            'pip', 'install', '--no-build-isolation', '.'],
            cwd=Path(__file__).parent.parent.parent.parent)
        self.click(By.XPATH, "//button[@aria-label='Close modal']")

        # launch the plugin
        self.click(By.NAME, "Test Plugin")
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((
                By.XPATH,
                "//div[contains(text(), 'Starting up model...')]")))

        # enter input data into the form
        WebDriverWait(self.driver, 60).until(
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
