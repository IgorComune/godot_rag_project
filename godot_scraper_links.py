#!/usr/bin/env python3
"""
Godot Documentation Scraper - Improved Version
- Saves each page individually as soon as it's scraped
- Better detection of deep sidebar hierarchy (up to 10+ levels)
- Two phases: Full discovery + Scraping with individual saving
"""

import requests
from bs4 import BeautifulSoup
import os
import json
import re
from urllib.parse import urljoin, urlparse, urlunparse
from collections import deque
import logging
from typing import Set, Dict, Optional
import time

class GodotDocsScraper:
    def __init__(self, base_url="https://docs.godotengine.org/en/stable/", 
                 output_dir="godot_docs", max_depth=10):
        self.base_url = self.normalize_url(base_url)
        self.output_dir = output_dir
        self.max_depth = max_depth
        
        # Phase 1: Mapping
        self.all_urls = set()
        self.url_to_depth = {}
        self.url_hierarchy = {}  # To track the complete hierarchy
        self.failed_discovery = set()
        
        # Phase 2: Scraping (without storing all content in memory)
        self.scraped_count = 0
        self.failed_scraping = set()
        
        # Output directories
        self.pages_dir = os.path.join(output_dir, "pages")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        
        # Shared session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.pages_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Improved patterns to avoid loops and unnecessary content
        self.skip_patterns = {
            'navigation': ['previous', 'next', 'prev', 'up', 'parent', 'edit page', 'edit on github'],
            'file_types': ['.pdf', '.zip', '.tar.gz', '.epub', '.png', '.jpg', '.gif', '.css', '.js', '.woff', '.ttf'],
            'system_pages': ['search.html', 'genindex.html', 'modindex.html', 'searchindex.js',
                             '_sources/', '_static/', '_images/', '_downloads/', 'searchindex.json'],
            'anchor_only': ['#', 'javascript:', 'mailto:', 'tel:', 'ftp://']
        }

    def normalize_url(self, url: str) -> str:
        """Robust URL normalization"""
        if not url:
            return ""
            
        parsed = urlparse(url)
        parsed = parsed._replace(fragment='', query='')
        
        path = parsed.path
        if path:
            # Resolve .. and . in the path
            path_parts = []
            for part in path.split('/'):
                if part == '..':
                    if path_parts:
                        path_parts.pop()
                elif part and part != '.':
                    path_parts.append(part)
            path = '/' + '/'.join(path_parts)
            
            if len(path) > 1 and path.endswith('/'):
                path = path.rstrip('/')
        else:
            path = '/'
            
        parsed = parsed._replace(path=path)
        return urlunparse(parsed)

    def is_valid_godot_doc_url(self, url: str) -> bool:
        """Enhanced validation for valid URLs"""
        if not url:
            return False
            
        parsed = urlparse(url)
        
        # Must be from the docs.godotengine.org domain
        if parsed.netloc != 'docs.godotengine.org':
            return False
            
        # Must have a valid version
        if not any(version in url for version in ['/en/stable/', '/en/latest/', '/en/4.', '/en/3.']):
            return False
            
        # Avoid file types
        if any(url.lower().endswith(ext) for ext in self.skip_patterns['file_types']):
            return False
            
        # Avoid system pages
        if any(pattern in url for pattern in self.skip_patterns['system_pages']):
            return False
            
        # Avoid anchor-only links
        if any(url.startswith(pattern) for pattern in self.skip_patterns['anchor_only']):
            return False
            
        return True

    def is_navigation_link(self, link_element, href: str) -> bool:
        """Detects navigation links that might cause loops"""
        if not link_element:
            return False
            
        # Element classes
        classes = link_element.get('class', [])
        if isinstance(classes, str):
            classes = [classes]
            
        nav_classes = ['prev', 'next', 'previous', 'up', 'parent', 'breadcrumb', 'headerlink', 'edit-on-github']
        if any(nav_class in ' '.join(classes).lower() for nav_class in nav_classes):
            return True
            
        # Link text
        link_text = link_element.get_text().lower().strip()
        if any(pattern in link_text for pattern in self.skip_patterns['navigation']):
            return True
            
        # Rel attribute
        rel = link_element.get('rel', [])
        if isinstance(rel, str):
            rel = [rel]
        if any(r in ['prev', 'next', 'previous', 'up'] for r in rel):
            return True
            
        # Parent with navigation classes
        parent = link_element.parent
        if parent:
            parent_classes = parent.get('class', [])
            if isinstance(parent_classes, str):
                parent_classes = [parent_classes]
            if any('nav' in cls.lower() or 'breadcrumb' in cls.lower() for cls in parent_classes):
                return True
                
        return False

    def get_page_content(self, url: str) -> Optional[str]:
        """Fetch page content with retry"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Error fetching {url} (attempt {attempt + 1}): {e}")
                    return None
                else:
                    self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
        return None

    def extract_links_from_page(self, soup: BeautifulSoup, base_url: str) -> Set[str]:
        """Extracts links with a focus on deep sidebar hierarchy"""
        links = set()
        
        # Expanded selectors to capture deep hierarchy
        sidebar_selectors = [
            # Main sidebar
            '.wy-nav-side .wy-menu-vertical',
            '.sphinxsidebarwrapper',
            '.sidebar-tree',
            
            # TOC (Table of Contents) - very important for hierarchy
            '.toctree-wrapper',
            '.toctree-wrapper ul',
            '.toctree-wrapper li',
            '.toctree-l1', '.toctree-l2', '.toctree-l3', '.toctree-l4', '.toctree-l5',
            '.toctree-l6', '.toctree-l7', '.toctree-l8', '.toctree-l9', '.toctree-l10',
            
            # Navigation menu
            'nav.main-navigation',
            '.main-navigation ul',
            
            # Links in the main content (to capture internal references)
            '.document .body a[href^="/"]',
            '.rst-content a[href^="/"]'
        ]
        
        # Search in all selectors
        for selector in sidebar_selectors:
            elements = soup.select(selector)
            for element in elements:
                # Find all links within the element
                for link in element.find_all('a', href=True):
                    href = link.get('href')
                    
                    if not href:
                        continue
                        
                    # Skip links that are just anchors or special
                    if any(href.startswith(pattern) for pattern in self.skip_patterns['anchor_only']):
                        continue
                        
                    # Skip navigation links
                    if self.is_navigation_link(link, href):
                        continue
                        
                    # Build full URL
                    full_url = urljoin(base_url, href)
                    
                    # Validate URL
                    if self.is_valid_godot_doc_url(full_url):
                        normalized_url = self.normalize_url(full_url)
                        links.add(normalized_url)
        
        return links

    def get_url_hierarchy_info(self, url: str) -> Dict:
        """Extracts hierarchy information from the URL"""
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split('/') if p and p not in ['en', 'stable', 'latest']]
        
        hierarchy_info = {
            'path_parts': path_parts,
            'depth_from_path': len(path_parts),
            'section': path_parts[0] if path_parts else 'root',
            'subsection': path_parts[1] if len(path_parts) > 1 else None,
            'filename': path_parts[-1] if path_parts else 'index'
        }
        
        return hierarchy_info

    # =========================================================================
    # PHASE 1: FULL DISCOVERY WITH DEEP HIERARCHY
    # =========================================================================
    
    def discover_all_pages(self):
        """PHASE 1: Discover all pages with deep hierarchy"""
        self.logger.info("🔍 PHASE 1: Discovering all pages with deep hierarchy...")
        
        queue = deque([(self.base_url, 0, [])])  # (url, depth, hierarchy_path)
        self.url_to_depth[self.base_url] = 0
        self.url_hierarchy[self.base_url] = []
        visited_discovery = set()
        
        depth_counts = {}  # For statistics
        
        while queue:
            url, depth, hierarchy_path = queue.popleft()
            url = self.normalize_url(url)
            
            if url in visited_discovery or depth > self.max_depth:
                continue
                
            # Depth statistics
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
            
            self.logger.info(f"🔍 [{depth_counts[depth]:4d}] D{depth:2d}: {url}")
            
            html = self.get_page_content(url)
            if not html:
                self.failed_discovery.add(url)
                continue
                
            visited_discovery.add(url)
            self.all_urls.add(url)
            self.url_to_depth[url] = depth
            self.url_hierarchy[url] = hierarchy_path
            
            # Find new links if we are not at maximum depth
            if depth < self.max_depth:
                soup = BeautifulSoup(html, 'html.parser')
                new_links = self.extract_links_from_page(soup, url)
                
                for link in new_links:
                    if link not in visited_discovery and link not in self.url_to_depth:
                        # Create hierarchy based on URL
                        hierarchy_info = self.get_url_hierarchy_info(link)
                        new_hierarchy = hierarchy_path + [hierarchy_info['section']]
                        
                        queue.append((link, depth + 1, new_hierarchy))
                        self.url_to_depth[link] = depth + 1
                        self.url_hierarchy[link] = new_hierarchy
        
        self.logger.info(f"✅ PHASE 1 COMPLETE:")
        self.logger.info(f"   📄 Total pages discovered: {len(self.all_urls)}")
        self.logger.info(f"   ❌ Discovery failures: {len(self.failed_discovery)}")
        
        # Show distribution by depth
        for depth in sorted(depth_counts.keys()):
            self.logger.info(f"   📊 Depth {depth}: {depth_counts[depth]} pages")
        
        # Save mapping
        self.save_page_mapping()

    def save_page_mapping(self):
        """Saves the complete mapping of discovered pages"""
        mapping_data = {
            'total_pages': len(self.all_urls),
            'discovery_timestamp': time.time(),
            'max_depth_found': max(self.url_to_depth.values()) if self.url_to_depth else 0,
            'pages_by_depth': {},
            'pages_by_section': {},
            'all_urls': sorted(list(self.all_urls)),
            'failed_discovery': list(self.failed_discovery),
            'url_metadata': {}
        }
        
        # Statistics by depth
        for url, depth in self.url_to_depth.items():
            if url in self.all_urls:
                mapping_data['pages_by_depth'][str(depth)] = mapping_data['pages_by_depth'].get(str(depth), 0) + 1
        
        # Statistics by section and detailed metadata
        for url in self.all_urls:
            hierarchy_info = self.get_url_hierarchy_info(url)
            section = hierarchy_info['section']
            
            mapping_data['pages_by_section'][section] = mapping_data['pages_by_section'].get(section, 0) + 1
            
            mapping_data['url_metadata'][url] = {
                'depth': self.url_to_depth.get(url, 0),
                'hierarchy': self.url_hierarchy.get(url, []),
                'section': section,
                'subsection': hierarchy_info['subsection'],
                'path_parts': hierarchy_info['path_parts']
            }
        
        mapping_file = os.path.join(self.metadata_dir, 'page_mapping.json')
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📋 Detailed mapping saved to: {mapping_file}")

    # =========================================================================
    # PHASE 2: SCRAPING WITH INDIVIDUAL SAVING
    # =========================================================================
    
    def scrape_all_content(self):
        """PHASE 2: Scraping with individual saving of each page"""
        total_pages = len(self.all_urls)
        self.logger.info(f"📄 PHASE 2: Scraping {total_pages} pages (individual saving)...")
        
        # Sort URLs by depth and section for organized scraping
        sorted_urls = sorted(self.all_urls, key=lambda url: (
            self.url_to_depth.get(url, 0),
            self.get_url_hierarchy_info(url)['section'],
            url
        ))
        
        for i, url in enumerate(sorted_urls, 1):
            # Detailed progress
            progress = (i / total_pages) * 100
            depth = self.url_to_depth.get(url, 0)
            section = self.get_url_hierarchy_info(url)['section']
            
            self.logger.info(f"📄 [{i:4d}/{total_pages}] ({progress:5.1f}%) [D{depth}] [{section}] {url}")
            
            # Scraping content
            success = self.scrape_and_save_page(url, i)
            if success:
                self.scraped_count += 1
            else:
                self.failed_scraping.add(url)
        
        self.logger.info(f"✅ PHASE 2 COMPLETE:")
        self.logger.info(f"   📄 Pages processed: {self.scraped_count}")
        self.logger.info(f"   ❌ Scraping failures: {len(self.failed_scraping)}")
        
        # Save final report
        self.save_final_report()

    def scrape_and_save_page(self, url: str, page_number: int) -> bool:
        """Scrapes a page and saves it immediately"""
        try:
            html = self.get_page_content(url)
            if not html:
                return False
            
            soup = BeautifulSoup(html, 'html.parser')
            content_data = self.extract_content(soup, url)
            
            # Save individual page
            self.save_individual_page(content_data, page_number)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {url}: {e}")
            return False

    def save_individual_page(self, content_data: Dict, page_number: int):
        """Saves an individual page"""
        url = content_data['url']
        hierarchy_info = self.get_url_hierarchy_info(url)
        
        # Create unique filename based on hierarchy
        section = hierarchy_info['section']
        filename_parts = [f"{page_number:04d}"]
        
        if section:
            filename_parts.append(section)
        
        if hierarchy_info['subsection']:
            filename_parts.append(hierarchy_info['subsection'])
            
        # Use path parts to create unique name
        path_part = '_'.join(hierarchy_info['path_parts'][-2:]) if len(hierarchy_info['path_parts']) > 1 else hierarchy_info['filename']
        filename_parts.append(path_part)
        
        # Sanitize filename
        filename = '_'.join(filename_parts)
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        filename = filename[:200]  # Limit size
        filename += '.json'
        
        # Save to appropriate directory
        section_dir = os.path.join(self.pages_dir, section)
        os.makedirs(section_dir, exist_ok=True)
        
        file_path = os.path.join(section_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(content_data, f, ensure_ascii=False, indent=2)

    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extracts enhanced content from a page"""
        hierarchy_info = self.get_url_hierarchy_info(url)
        
        content_data = {
            'url': url,
            'scraping_timestamp': time.time(),
            'title': '',
            'content': '',
            'headings': [],
            'code_blocks': [],
            'links': [],
            'images': [],
            'tables': [],
            'section': hierarchy_info['section'],
            'subsection': hierarchy_info['subsection'],
            'depth': self.url_to_depth.get(url, 0),
            'hierarchy': self.url_hierarchy.get(url, []),
            'path_parts': hierarchy_info['path_parts']
        }
        
        # Page title
        title_selectors = ['title', 'h1', '.document h1', '.rst-content h1']
        for selector in title_selectors:
            title_element = soup.select_one(selector)
            if title_element:
                content_data['title'] = title_element.get_text().strip()
                break
        
        # Main content
        main_selectors = [
            '.document .body',
            '.rst-content .document',
            '.rst-content [role="main"]',
            '.content',
            'main',
            '[role="main"]',
            '.main-content'
        ]
        
        main_content = None
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if main_content:
            # Clean unnecessary elements
            for tag in main_content.find_all(['script', 'style', 'nav', '.headerlink', '.edit-on-github']):
                tag.decompose()
            
            # Extract headings with hierarchy
            content_data['headings'] = []
            for h in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                content_data['headings'].append({
                    'level': int(h.name[1]),
                    'text': h.get_text().strip(),
                    'id': h.get('id', ''),
                    'classes': h.get('class', [])
                })
            
            # Extract code blocks with context
            content_data['code_blocks'] = []
            for code_element in main_content.find_all(['code', 'pre']):
                code_text = code_element.get_text().strip()
                if len(code_text) > 5:  # Only significant codes
                    content_data['code_blocks'].append({
                        'language': self.detect_language(code_element),
                        'code': code_text,
                        'classes': code_element.get('class', []),
                        'parent_tag': code_element.parent.name if code_element.parent else ''
                    })
            
            # Extract internal links
            content_data['links'] = []
            for link in main_content.find_all('a', href=True):
                href = link.get('href')
                if href and not any(href.startswith(p) for p in self.skip_patterns['anchor_only']):
                    full_url = urljoin(url, href)
                    if self.is_valid_godot_doc_url(full_url):
                        content_data['links'].append({
                            'text': link.get_text().strip(),
                            'href': href,
                            'full_url': self.normalize_url(full_url)
                        })
            
            # Extract images
            content_data['images'] = []
            for img in main_content.find_all('img', src=True):
                content_data['images'].append({
                    'src': img.get('src'),
                    'alt': img.get('alt', ''),
                    'title': img.get('title', '')
                })
            
            # Extract tables
            content_data['tables'] = []
            for table in main_content.find_all('table'):
                headers = [th.get_text().strip() for th in table.find_all('th')]
                rows = []
                for tr in table.find_all('tr'):
                    row = [td.get_text().strip() for td in tr.find_all(['td', 'th'])]
                    if row:
                        rows.append(row)
                
                if headers or rows:
                    content_data['tables'].append({
                        'headers': headers,
                        'rows': rows
                    })
            
            # Clean main text content
            content_data['content'] = main_content.get_text().strip()
        
        return content_data

    def detect_language(self, code_element) -> str:
        """Enhanced programming language detection"""
        # Element classes
        classes = code_element.get('class', [])
        if isinstance(classes, str):
            classes = [classes]
        
        # Check for specific language classes
        language_classes = ['gdscript', 'csharp', 'cpp', 'python', 'json', 'ini', 'bash', 'shell', 'xml', 'html', 'css', 'javascript']
        for cls in classes:
            if cls.startswith('language-'):
                return cls.replace('language-', '')
            elif cls.startswith('highlight-'):
                return cls.replace('highlight-', '')
            elif cls in language_classes:
                return cls
        
        # Try to detect by code structure
        code_text = code_element.get_text().strip()
        if code_text:
            # GDScript
            if any(keyword in code_text for keyword in ['extends ', 'func ', 'var ', 'export ', 'onready ']):
                return 'gdscript'
            # C#
            elif any(keyword in code_text for keyword in ['using ', 'namespace ', 'public class ', 'private void ']):
                return 'csharp'
            # JSON
            elif code_text.startswith(('{', '[')) and code_text.endswith(('}', ']')):
                return 'json'
        
        return 'text'

    def save_final_report(self):
        """Saves a detailed final report"""
        report = {
            'scraping_summary': {
                'total_discovered': len(self.all_urls),
                'total_scraped': self.scraped_count,
                'failed_discovery': len(self.failed_discovery),
                'failed_scraping': len(self.failed_scraping),
                'success_rate': (self.scraped_count / len(self.all_urls)) * 100 if self.all_urls else 0,
                'scraping_timestamp': time.time()
            },
            'sections_scraped': self.get_section_stats(),
            'depth_distribution': self.get_depth_stats(),
            'file_organization': self.get_file_organization_info(),
            'failed_urls': {
                'discovery': list(self.failed_discovery),
                'scraping': list(self.failed_scraping)
            }
        }
        
        report_file = os.path.join(self.metadata_dir, 'scraping_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📊 Final report saved to: {report_file}")

    def get_section_stats(self) -> Dict[str, int]:
        """Statistics by section of processed pages"""
        stats = {}
        for url in self.all_urls:
            if url not in self.failed_scraping:
                section = self.get_url_hierarchy_info(url)['section']
                stats[section] = stats.get(section, 0) + 1
        return stats

    def get_depth_stats(self) -> Dict[str, int]:
        """Statistics by depth of processed pages"""
        stats = {}
        for url in self.all_urls:
            if url not in self.failed_scraping:
                depth = str(self.url_to_depth.get(url, 0))
                stats[depth] = stats.get(depth, 0) + 1
        return stats

    def get_file_organization_info(self) -> Dict:
        """Information about the organization of saved files"""
        info = {
            'pages_directory': self.pages_dir,
            'metadata_directory': self.metadata_dir,
            'sections_directories': []
        }
        
        # List created section directories
        if os.path.exists(self.pages_dir):
            for item in os.listdir(self.pages_dir):
                section_path = os.path.join(self.pages_dir, item)
                if os.path.isdir(section_path):
                    file_count = len([f for f in os.listdir(section_path) if f.endswith('.json')])
                    info['sections_directories'].append({
                        'section': item,
                        'file_count': file_count,
                        'path': section_path
                    })
        
        return info

    def run_full_scraping(self):
        """Executes the complete 2-phase scraping process"""
        try:
            start_time = time.time()
            
            # PHASE 1: Full discovery
            self.discover_all_pages()
            
            # PHASE 2: Scraping with individual saving
            self.scrape_all_content()
            
            total_time = time.time() - start_time
            self.logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during scraping: {e}")
            # Save report even in case of error
            try:
                self.save_final_report()
            except:
                pass
            return False

def main():
    """Main function - runs the scraper with optimized settings"""
    print("🚀 Godot Documentation Scraper - Improved Version")
    print("=" * 60)
    
    # Scraper settings
    base_url = "https://docs.godotengine.org/en/stable/"
    output_dir = "godot_docs_scraped"
    max_depth = 8  # Increased to capture deep hierarchy
    
    print(f"📁 Base URL: {base_url}")
    print(f"💾 Output directory: {output_dir}")
    print(f"🔢 Maximum depth: {max_depth}")
    print("=" * 60)
    
    # Create and run the scraper
    scraper = GodotDocsScraper(
        base_url=base_url,
        output_dir=output_dir,
        max_depth=max_depth
    )
    
    # Execute the full process
    success = scraper.run_full_scraping()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ SCRAPING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📄 Total pages processed: {scraper.scraped_count}")
        print(f"📁 Files saved in: {output_dir}")
        print(f"📊 Reports available in: {scraper.metadata_dir}")
        
        # Show final statistics
        if scraper.scraped_count > 0:
            print(f"📈 Success rate: {(scraper.scraped_count / len(scraper.all_urls)) * 100:.1f}%")
        
        # List processed sections
        section_stats = scraper.get_section_stats()
        if section_stats:
            print("\n📋 Pages by section:")
            for section, count in sorted(section_stats.items()):
                print(f"   • {section}: {count} pages")
        
        print("\n🎉 Godot documentation successfully downloaded and organized!")
        
    else:
        print("\n" + "=" * 60)
        print("❌ ERROR DURING SCRAPING")
        print("=" * 60)
        print("Check logs for more details on errors.")
        if scraper.failed_discovery:
            print(f"❌ Discovery failures: {len(scraper.failed_discovery)}")
        if scraper.failed_scraping:
            print(f"❌ Scraping failures: {len(scraper.failed_scraping)}")
    
    print(f"\n📁 All files have been saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()