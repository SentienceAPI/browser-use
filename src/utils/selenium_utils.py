from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


def setup_selenium_driver(headless: bool = False) -> webdriver.Chrome:
    """
    Sets up and returns a Selenium WebDriver instance.

    Args:
        headless (bool): Whether to run browser in headless mode

    Returns:
        webdriver.Chrome: Configured Chrome WebDriver instance
    """
    # Configure Chrome options
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")

    # Disable automation flags
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    # Initialize the Chrome driver
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    # Add cookie acceptance function
    def accept_cookies(driver):
        """
        More robust cookie acceptance function that tries multiple approaches
        """
        try:
            # Common cookie accept button selectors
            cookie_selectors = [
                "button[id*='cookie-accept']",
                "button[id*='accept-cookie']",
                "button[class*='cookie-accept']",
                "button[class*='accept-cookie']",
                "[aria-label*='accept' i][role='button']",
                "[aria-label*='cookie' i][role='button']",
                "button:contains('Accept')",
                "button:contains('Akzeptieren')",
                "button:contains('Zustimmen')",
                "button:contains('Alle akzeptieren')",
                "#onetrust-accept-btn-handler",
                "#accept-all-cookies",
                ".accept-cookies",
                ".accept-all-cookies"
            ]

            wait = WebDriverWait(driver, 10)

            # Try each selector
            for selector in cookie_selectors:
                try:
                    # Try to find element by CSS selector
                    cookie_button = wait.until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    cookie_button.click()
                    return True
                except:
                    try:
                        # Try to find element by XPath containing text
                        xpath = f"//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{selector.lower()}')]"
                        cookie_button = wait.until(
                            EC.element_to_be_clickable((By.XPATH, xpath))
                        )
                        cookie_button.click()
                        return True
                    except:
                        continue

            # Try finding iframe with cookie consent
            iframes = driver.find_elements(By.TAG_NAME, "iframe")
            for iframe in iframes:
                try:
                    driver.switch_to.frame(iframe)
                    for selector in cookie_selectors:
                        try:
                            cookie_button = wait.until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                            )
                            cookie_button.click()
                            driver.switch_to.default_content()
                            return True
                        except:
                            continue
                    driver.switch_to.default_content()
                except:
                    driver.switch_to.default_content()
                    continue

        except Exception as e:
            print(f"Error accepting cookies: {str(e)}")
        return False

    # Attach the function to the driver instance
    driver.accept_cookies = lambda: accept_cookies(driver)

    return driver
