import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
import os
import pandas as pd
from typing import Optional, Dict
import hashlib


class WebContentExtractor:
    """
    Class to extract specific content from web pages and save it to text files.
    """

    def __init__(self, timeout: int = 10):
        """
        Initializes the extractor with default settings.

        Args:
            timeout (int): Timeout for HTTP requests in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        # Headers to simulate a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def _get_page_name(self, url: str) -> str:
        """
        Extracts the page name from the URL.

        Args:
            url (str): Page URL

        Returns:
            str: Page name formatted for filename
        """
        parsed_url = urlparse(url)
        path_parts = [part for part in parsed_url.path.split('/') if part]

        if path_parts:
            page_name = path_parts[-1]
            page_name = re.sub(r'\.(html|htm|php|asp|aspx)$', '', page_name, flags=re.IGNORECASE)
        else:
            page_name = parsed_url.netloc.replace('www.', '')

        page_name = re.sub(r'[<>:"/\\|?*]', '_', page_name)
        if not page_name:
            page_name = 'page'
        return page_name

    def _fetch_page(self, url: str) -> str:
        """
        Sends an HTTP request and returns the HTML content.

        Args:
            url (str): Page URL

        Returns:
            str: HTML content of the page

        Raises:
            requests.RequestException: If there's a request error
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise requests.RequestException(f"Error accessing URL {url}: {str(e)}")

    def _extract_content(self, html: str) -> str:
        """
        Extracts content from a specific element.

        Args:
            html (str): HTML content of the page

        Returns:
            str: Extracted text from the element

        Raises:
            ValueError: If the element is not found
        """
        soup = BeautifulSoup(html, 'html.parser')

        target_element = soup.find('div', {
            'role': 'main',
            'class': 'document',
            'itemscope': 'itemscope',
            'itemtype': 'http://schema.org/Article'
        })

        if not target_element:
            raise ValueError("Element with the specified attributes was not found on the page")

        text = target_element.get_text(separator='\n', strip=True)
        return text

    def _save_to_file(self, content: str, filename: str, output_dir: str = '.') -> str:
        """
        Saves the content to a text file.

        Args:
            content (str): Content to be saved
            filename (str): File name (without extension)
            output_dir (str): Output directory

        Returns:
            str: Full path of the saved file
        """
        os.makedirs(output_dir, exist_ok=True)

        safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
        url_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()
        final_filename = f"{safe_filename[:150]}_{url_hash}"

        filepath = os.path.join(output_dir, f"{final_filename}.txt")

        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)

        return filepath

    def extract_and_save(self, url: str, output_dir: str = '.', custom_filename: Optional[str] = None) -> Dict:
        """
        Extracts content from a URL and saves it to a file.

        Args:
            url (str): Page URL to be processed
            output_dir (str): Directory to save the file (default: current directory)
            custom_filename (str, optional): Custom file name (without extension).
                                             If not provided, the URL will be used.

        Returns:
            dict: Processing information

        Raises:
            requests.RequestException: If there's an HTTP request error
            ValueError: If the element is not found
            Exception: For other errors during processing
        """
        try:
            html_content = self._fetch_page(url)
            extracted_text = self._extract_content(html_content)

            filename_to_use = custom_filename if custom_filename else url
            filepath = self._save_to_file(extracted_text, filename_to_use, output_dir)

            return {
                'success': True,
                'url': url,
                'filename': filename_to_use,
                'filepath': filepath,
                'content_length': len(extracted_text),
                'message': f'Content successfully extracted and saved to: {filepath}'
            }

        except Exception as e:
            return {
                'success': False,
                'url': url,
                'error': str(e),
                'message': f'Error processing URL: {str(e)}'
            }

    def process_csv_links(self, csv_path: str = "godot_docs/godot_docs_links.csv",
                          output_dir: str = "godot_docs/pages") -> Dict:
        """
        Processes all links from a CSV file automatically.

        Args:
            csv_path (str): Path to the CSV file containing the links
            output_dir (str): Directory to save the extracted files

        Returns:
            dict: Processing report with statistics
        """
        try:
            print(f"📂 Loading CSV file: {csv_path}")
            df = pd.read_csv(csv_path)

            if 'link' not in df.columns:
                raise ValueError("'link' column not found in CSV")

            links = df['link'].dropna().tolist()
            total_links = len(links)

            print(f"🔗 Found {total_links} links to process\n")

            successful = 0
            failed = 0
            failed_links = []

            for i, link in enumerate(links, 1):
                print(f"🔄 Processing {i}/{total_links}: {link}")

                result = self.extract_and_save(link, output_dir)

                if result['success']:
                    successful += 1
                    print(f"   ✅ Saved: {result['filepath']}")
                else:
                    failed += 1
                    failed_links.append({'link': link, 'error': result['error']})
                    print(f"   ❌ Error: {result['error']}")

                print()

            print("=" * 50)
            print("📊 FINAL REPORT")
            print("=" * 50)
            print(f"Total processed: {total_links}")
            print(f"Successes: {successful}")
            print(f"Failures: {failed}")
            print(f"Success rate: {(successful/total_links)*100:.1f}%")

            if failed_links:
                print(f"\n❌ Failed links:")
                for item in failed_links:
                    print(f"   - {item['link']}: {item['error']}")

            return {
                'total_processed': total_links,
                'successful': successful,
                'failed': failed,
                'failed_links': failed_links,
                'success_rate': (successful/total_links)*100 if total_links > 0 else 0
            }

        except Exception as e:
            print(f"❌ Error processing CSV: {str(e)}")
            return {
                'total_processed': 0,
                'successful': 0,
                'failed': 0,
                'failed_links': [],
                'success_rate': 0,
                'error': str(e)
            }

# Example usage
if __name__ == "__main__":
    extractor = WebContentExtractor()

    print("🚀 Starting processing of Godot Docs links...")
    result = extractor.process_csv_links()
