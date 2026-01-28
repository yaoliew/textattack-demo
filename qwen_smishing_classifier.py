import os

# load in environment variables
from dotenv import load_dotenv
load_dotenv()

import json
import requests
import base64
import subprocess
import whois
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from requests.structures import CaseInsensitiveDict

import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from textattack.models.wrappers import ModelWrapper


class QwenSmishingClassifier(ModelWrapper):
    """
    TextAttack-compatible wrapper for Qwen2.5-VL-7B-Instruct smishing/phishing classification.
    
    Uses single-token label mapping to extract logits for "Legitimate" and "Smishing"
    tokens for binary smishing detection (0 = Legitimate, 1 = Smishing/Spam).
    """
    
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", device=None):
        """
        Initialize the Qwen smishing classifier.
        
        Args:
            model_name: Hugging Face model identifier
            device: torch device (auto-detected if None)
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Primary: Read API keys from environment variables
        self.jina_api_key = os.getenv('JINA_API_KEY')
        self.google_cloud_API_key = os.getenv('GOOGLE_CLOUD_API_KEY')
        self.search_engine_ID = os.getenv('SEARCH_ENGINE_ID')
        
        # Fallback: Try to import from config file (for development/testing only)
        if not self.jina_api_key:
            try:
                from demos.prompting_model.config import jina_api_key as config_jina_key
                self.jina_api_key = config_jina_key
            except ImportError:
                pass  # Will raise error if needed later when making API calls
        
        if not self.google_cloud_API_key:
            try:
                from demos.prompting_model.config import google_cloud_API_key as config_google_key
                self.google_cloud_API_key = config_google_key
            except ImportError:
                pass
        
        if not self.search_engine_ID:
            try:
                from demos.prompting_model.config import search_engine_ID as config_search_id
                self.search_engine_ID = config_search_id
            except ImportError:
                pass
        
        # HTTP request header (from config or default)
        try:
            from demos.prompting_model.config import http_request_header as config_header
            self.http_request_header = config_header
        except ImportError:
            self.http_request_header = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }
        
        print(f"Loading Qwen model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load processor and model
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="cuda:0" if self.device.type == "cuda" else None,
            )
            
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

        # Expose processor as tokenizer for attack recipes
        self.tokenizer = self.processor.tokenizer

        # Get token IDs for label tokens
        self.legitimate_token_id, self.smishing_token_id = self._get_label_token_ids()
        print(f"Label token IDs - Legitimate: {self.legitimate_token_id}, Smishing: {self.smishing_token_id}")
    
    def _get_label_token_ids(self):
        """
        Find token IDs for "Legitimate" and "Smishing" labels.
        Handles single-token and multi-token cases.
        
        Returns:
            tuple: (legitimate_token_id, smishing_token_id)
        """
        # Try to get single tokens first
        legitimate_tokens = self.processor.tokenizer.encode("Legitimate", add_special_tokens=False)
        smishing_tokens = self.processor.tokenizer.encode("Smishing", add_special_tokens=False)
        
        # Use first token if multi-token, or the single token
        legitimate_token_id = legitimate_tokens[0] if legitimate_tokens else None
        smishing_token_id = smishing_tokens[0] if smishing_tokens else None
        
        # Verify tokens exist in vocabulary
        if legitimate_token_id is None or smishing_token_id is None:
            raise ValueError("Could not find token IDs for 'Legitimate' and 'Smishing' labels")
        
        # Check if tokens are the same (shouldn't happen, but safety check)
        if legitimate_token_id == smishing_token_id:
            raise ValueError("'Legitimate' and 'Smishing' map to the same token ID")
        
        return legitimate_token_id, smishing_token_id
    
    def _call_qwen_for_json(self, prompt: str) -> Dict:
        """
        Call Qwen and parse JSON from text response.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Dict: Parsed JSON response
        """
        # Format using chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        formatted_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = self.processor.tokenizer([formatted_text], return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
            )
        
        # Extract only the newly generated tokens (not the input)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = generated_ids[0][input_length:]
        
        # Decode only the generated tokens
        response_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Clean and parse JSON
        cleaned_response = self._clean_json_response(response_text)
        
        # Try to parse JSON, with better error handling
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            # Log the problematic response for debugging
            print(f"JSON decode error. Response text: {response_text[:200]}...")
            print(f"Cleaned response: {cleaned_response[:200]}...")
            raise ValueError(f"Failed to parse JSON response: {e}. Response: {response_text[:500]}")
    
    def _call_qwen_for_text(self, prompt: str) -> str:
        """
        Call Qwen for text responses.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            str: Text response
        """
        # Format using chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        formatted_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = self.processor.tokenizer([formatted_text], return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
            )
        
        # Extract only the newly generated tokens (not the input)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = generated_ids[0][input_length:]
        
        # Decode only the generated tokens
        response_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return response_text
    
    def _call_qwen_vision(self, image_path: str, text_prompt: str) -> str:
        """
        Call Qwen2.5-VL for image analysis.
        
        Args:
            image_path: Path to image file
            text_prompt: Text prompt for analysis
            
        Returns:
            str: Analysis result
        """
        from PIL import Image
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Format using chat template with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]
        
        # Process inputs
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                temperature=None,
            )
        
        # Extract only the newly generated tokens (not the input)
        # For vision inputs, input_ids contains the tokenized text + image tokens
        input_length = inputs.input_ids.shape[1]
        generated_tokens = generated_ids[0][input_length:]
        
        # Decode only the generated tokens
        response_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return response_text
    
    def _clean_json_response(self, content: str) -> str:
        """Clean GPT response to extract valid JSON."""
        import re
        
        # Remove markdown code blocks
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        # Try to find JSON object boundaries
        # Look for first { and last }
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            # Extract JSON object
            content = content[first_brace:last_brace + 1]
        
        # Remove leading/trailing whitespace and newlines
        content = content.strip()
        
        # Replace newlines within the JSON (but preserve structure)
        # Only replace newlines that are not part of string values
        # This is a simple approach - for complex cases, we might need a more sophisticated parser
        content = re.sub(r'\n\s*', ' ', content)
        
        return content
    
    def _prepare_for_json_serialization(self, data):
        """Prepare data for JSON serialization by handling special types."""
        if isinstance(data, CaseInsensitiveDict):
            return dict(data)
        elif isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, set):
            return list(data)
        elif isinstance(data, dict):
            return {k: self._prepare_for_json_serialization(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json_serialization(item) for item in data]
        else:
            return data
    
    def _get_url_extraction_prompt(self) -> str:
        """Get prompt for URL and brand extraction."""
        return """Extract any URLs, and brand names from the following SMS message.
        Your output should be in json format and should not have any other output: 
        - is_URL: true or false
        - URLs: if no URL in SMS, answer non. If there are URLs, the response should be a list, each element is a URL extracted from the SMS. 
        - is_brand: true or false
        - brands: if no brand name in SMS, answer non. If there are brand names, the response should be a list, each element is a brand name extracted from the SMS. You can extract the brand name from the SMS content and the URL."""
    
    def _get_detection_prompt_template(self) -> str:
        """Get the main detection prompt template."""
        return """I want you to act as a spam detector to determine whether a given SMS is phishing, spam, or legitimate. Your analysis should be thorough and evidence-based. Analyze the SMS by following these steps:
        1. If the SMS is promoting any of the following categories: Online gambling, bets, spins, adult content, digital currency, lottery, it is either spam or phishing.
        2. The SMS is legitimate if it is from known organizations, such as appointment reminders,  OTP (One-Time Password) verification, delivery notifications, account updates, tracking information, or other expected messages.
        3. The SMS is considered legitimate if it involves a conversation between friends, family members, or colleagues.
        4. Promotions and advertisements are considered spam. The SMS is spam if it is promotion from legitimate companies and is not impersonating any brand, but it is advertisements, app download promotions, sales promotions, donation requests, event promotions, online loan services, or other irrelevant information.
        5. The SMS is phishing if it is fraudulent and attempts to deceive recipients into providing sensitive information or clicking malicious links. Phishing SMS may exhibit the following characteristics:
        Promotions or Rewards: Some phishing SMS offer fake prizes, rewards, or other incentives to lure recipients into clicking links or providing personal information.
        Urgent or Alarming Language: Phishing messages often create a sense of urgency or fear, such as threats of account suspension, missed payments, or urgent security alerts.
        Suspicious Links: Phishing messages may contain links to fake websites designed to steal personal information.
        Requests for Personal Information: Phishing SMS may ask for sensitive information like passwords, credit card numbers, social security numbers, or other personal details.
        Grammatical and Spelling Errors: Many phishing messages contain grammatical mistakes or unusual wording, which can be a red flag for recipients.
        Expired Domain: Phishing websites often use domains that expire quickly or are already listed for sale.
        Inconsistency: The URL may be irrelevant to the message content.
        6. Please be aware that: It is common to see shortened URLs in SMS. You can get the expanded URL from the provided redirection chain. Both phishing and legitimate URLs can be shortened. And both phishing and legitimate websites may use a robot-human verification page (CAPTCHA-like mechanism) before granting access the content.
        7. I will provide you with some external information if there is a URL in the SMS. The information includes:
        - Redirect Chain: The URL may redirect through multiple intermediate links before reaching the final destination; if any of them is flagged as phishing, the original URL becomes suspicious.
        - Brand Search Information:  The top five results from a Google search of the brand name. You can compare if the URL's domain matches the results from Google.
        - Screenshot Description: A description of the website's screenshot, highlighting any notable visual elements.
        - HTML Content Summary: The title of HTML, and the summary of its content.
        - Domain Information: The domain registration details, including registrar, creation date, and DNS records, which are analyzed to verify the domain's legitimacy.
        8. Please give your rationales before making a decision. And your output should be in json format and should not have any other output:
        - brand\_impersonated: brand name associated with the SMS, if applicable.
        - URL: any URL appears in SMS, if no URL, answer "non".
        - rationales: detailed rationales for the determination, up to 500 words. Directly give sentences, do not categorize the rationales. Only tell the reasons why the SMS is legitimate or not, do not include the reasons why the SMS is spam or phishing.
        - brief\_reason: brief reason for the determination.
        - category: True or False. If the SMS is legitimate, output False. Else, output True.
        - advice: If the SMS is phishing, output potential risk and your advice for the recipients, such as ''Do not respond to this message or access the link.''

        Below is the information of the SMS:"""
    
    def _get_user_friendly_prompt(self) -> str:
        """Get prompt for generating user-friendly output."""
        return """Based on the detailed analysis, I want you to create a simple and easy-to-understand response to tell the user whether the text message is a phishing attempt or a legitimate message. Use plain language and avoid technical terms like URL or HTTP headers. Explain your conclusion in 3 sentences, focusing on whether the message seems suspicious or safe. Provide a simple reason to support your conclusion, including clear evidence such as a suspicious website link or an urgent tone in the message. The response should be reassuring and concise, easy for anyone to understand."""
    
    def _format_smishing_prompt(self, text):
        """
        Format a text input into a smishing classification prompt.
        
        Args:
            text: Input text to classify
            
        Returns:
            str: Formatted prompt string
        """
        prompt = f"Classify the following message as either Legitimate or Smishing (phishing/spam).\n\nMessage: {text}\n\nClassification:"
        return prompt
    
    def _extract_urls_and_brands(self, sms_message: str) -> Dict:
        """Extract URLs and brand names from SMS message."""
        prompt = self._get_url_extraction_prompt() + "\n" + sms_message
        
        try:
            response = self._call_qwen_for_json(prompt)
            return response
        except Exception as e:
            print(f"URL/brand extraction failed: {e}")
            return {"is_URL": False, "URLs": "non", "is_brand": False, "brands": "non"}
    
    def _normalize_url(self, url: str) -> str:
        """Add http:// prefix if missing."""
        if not (url.startswith("http://") or url.startswith("https://")):
            return "http://" + url
        return url
    
    def _check_url_validity(self, url: str) -> Tuple[bool, Optional[int]]:
        """Check if URL is valid and accessible."""
        try:
            response = requests.head(url, allow_redirects=True, headers=self.http_request_header)

            # If the status code is in the range of 200 to 399, the URL is valid
            if response.status_code in range(200, 400):
                return True, response.status_code
            else:
                return False, response.status_code
        except requests.exceptions.RequestException as e:
            print(f"Error occurred: {e}")
            return False, None
    
    def _expand_url(self, url: str) -> Optional[str]:
        """Expand shortened URLs to their final destination."""
        try:
            response = requests.head(url, allow_redirects=True, headers=self.http_request_header, timeout=10)
            return response.url
        except requests.RequestException:
            return None
    
    def _get_redirect_chain(self, url: str) -> List[Tuple[str, int]]:
        """Get the complete redirect chain for a URL."""
        try:
            response = requests.head(url, allow_redirects=True, headers=self.http_request_header)

            response_chain = []
            response_status = []

            if response.history:
                for resp in response.history:
                    response_chain.append(resp.url)
                    response_status.append(resp.status_code)

            # Add the final response URL and status
            response_chain.append(response.url)
            response_status.append(response.status_code)

            return list(zip(response_chain, response_status))
        except requests.RequestException:
            return "non"
    
    def _analyze_html_content(self, url: str) -> Tuple[str, str]:
        """Analyze HTML content of the URL."""
        if not self.jina_api_key:
            raise ValueError("Jina API key is required for HTML content analysis. Set JINA_API_KEY environment variable.")
        
        try:
            jina_url = f'https://r.jina.ai/{url}'
            headers = {"Authorization": f"Bearer {self.jina_api_key}"}
            response = requests.get(jina_url, headers=headers)
            
            # Limit content length
            content = response.text[:10000] if len(response.text) > 10000 else response.text
            
            # Summarize content using Qwen
            summary = self._summarize_html_content(content)
            return content, summary
            
        except requests.RequestException:
            content = "There is no information known about the URL. The URL might be invalid or expired."
            return content, content
    
    def _summarize_html_content(self, content: str) -> str:
        """Summarize HTML content using Qwen."""
        prompt = """Please summarize the content in English and determine whether the website has a block wall or not.
        Your output should be in json format and should not have any other output:
        - summary: the summary of the content in English. Within 500 words. Some website might have a robot-human verification page. If the website has no information available, mention that the content might be hidden behind a verification wall. Both phishing and legitimate websites can have a robot-human verification page. It doesn't necessarily indicate malicious intent.
        """ + f"\n\nThe website content: {content}"
        
        try:
            response = self._call_qwen_for_json(prompt)
            return response.get('summary', 'Could not analyze content')
        except Exception:
            return "Could not analyze content"
    
    def _get_domain_info(self, url: str) -> str:
        """Get domain registration information."""
        try:
            domain = url.split("//")[-1].split("/")[0]
            domain_info = whois.whois(domain)
            return str(domain_info)
        except Exception:
            return "non"
    
    def _take_screenshot(self, url: str, screenshot_path: str):
        """Take screenshot using Node.js crawler."""
        try:
            # Find crawler.js - check in demos/prompting_model and prompting_model directories
            crawler_paths = [
                'demos/prompting_model/crawler.js',
                'prompting_model/crawler.js',
                'crawler.js'
            ]
            
            crawler_js = None
            for path in crawler_paths:
                if os.path.exists(path):
                    crawler_js = path
                    break
            
            if not crawler_js:
                raise FileNotFoundError("crawler.js not found. Please ensure it exists in the project.")
            
            subprocess.run(
                ['node', crawler_js, url, screenshot_path],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"Screenshot saved to {screenshot_path}")
        except subprocess.SubprocessError as e:
            print(f"Screenshot capture failed: {e}")
            raise
    
    def _analyze_screenshot_with_qwen(self, image_path: str) -> str:
        """Analyze screenshot using Qwen Vision."""
        try:
            prompt = """You are a website screenshot analysis assistant. Your primary function is to analyze website screenshots, provide a detailed description of the content, and determine the purpose of the page. For instance:
                            If the screenshot shows a news site, summarize the main news topics or articles.
                            Identify any logos, brands, or key visual elements.
                            The URL might be redirected to a robot-human verification page. If the screenshot is a blank page, mention that the content might be hidden behind a verification wall.
                            Your response should be in English and plain text, without any markdown or HTML formatting. Your response should be in 15 sentences or less."""
            
            return self._call_qwen_vision(image_path, prompt)
            
        except Exception as e:
            print(f"Screenshot analysis failed: {e}")
            return "non"
    
    def _analyze_screenshot(self, url: str, output_dir: str, idx: int) -> Tuple[str, str]:
        """Take and analyze screenshot of the webpage."""
        try:
            screenshot_path = os.path.join(output_dir, f"screenshot_{idx}.png")
            
            # Take screenshot if it doesn't exist
            if not os.path.exists(screenshot_path):
                os.makedirs(output_dir, exist_ok=True)
                self._take_screenshot(url, screenshot_path)
            
            # Analyze screenshot with Qwen Vision
            image_content = self._analyze_screenshot_with_qwen(screenshot_path)
            return screenshot_path, image_content
            
        except Exception as e:
            print(f"Screenshot error: {e}")
            return "non", "non"
    
    def _google_search_brand(self, brand_name: str) -> List[str]:
        """Search Google for brand's official domains."""
        if not self.google_cloud_API_key or not self.search_engine_ID:
            raise ValueError("Google Cloud API key and Search Engine ID are required. Set GOOGLE_CLOUD_API_KEY and SEARCH_ENGINE_ID environment variables.")
        
        try:
            url = f"https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_cloud_API_key,
                'cx': self.search_engine_ID,
                'q': brand_name,
                'num': 5,  # Get top 5 results
            }
            response = requests.get(url, params=params)
            response = response.json()
            return [item['link'] for item in response.get('items', [])]
        except Exception as e:
            print(f"Google search failed: {e}")
            return []
    
    def _search_brand_domains(self, brands: List[str]) -> Dict:
        """Search for official domains of mentioned brands."""
        brand_search = {}
        
        for idx, brand in enumerate(brands):
            brand_search[idx] = {
                'brand_name': brand,
                'brand_domain': self._google_search_brand(brand_name=brand)
            }
        
        return brand_search
    
    def _analyze_urls(
        self, 
        urls: List[str], 
        output_dir: str,
        enable_redirect_chain: bool = True,
        enable_brand_search: bool = True,
        enable_screenshot: bool = True,
        enable_html_content: bool = True,
        enable_domain_info: bool = True
    ) -> Dict:
        """Analyze each URL in the SMS message."""
        url_analysis = {}
        
        for idx, url in enumerate(urls):
            url_analysis[idx] = {'URL': url}
            
            # Normalize URL
            normalized_url = self._normalize_url(url)
            final_url = self._expand_url(normalized_url)
            
            url_analysis[idx]['final_URL'] = final_url or normalized_url
            
            # Redirect chain analysis
            if enable_redirect_chain:
                redirect_chain = self._get_redirect_chain(normalized_url)
                url_analysis[idx]['redirect_chain'] = redirect_chain
            
            # HTML content analysis
            if enable_html_content:
                html_content, html_summary = self._analyze_html_content(final_url or normalized_url)
                url_analysis[idx]['URL_content'] = html_content
                url_analysis[idx]['html_summary'] = html_summary
            
            # Domain information
            if enable_domain_info:
                domain_info = self._get_domain_info(normalized_url)
                url_analysis[idx]['domain_info'] = domain_info
            
            # Screenshot analysis
            if enable_screenshot:
                screenshot_path, image_content = self._analyze_screenshot(
                    final_url or normalized_url, 
                    output_dir, 
                    idx
                )
                url_analysis[idx]['screenshot_path'] = screenshot_path
                url_analysis[idx]['Image_content'] = image_content
        
        return url_analysis
    
    def _build_detection_prompt(self, sms_message: str, analysis: Dict) -> str:
        """Build the comprehensive prompt for final detection."""
        prompt = self._get_detection_prompt_template()
        prompt += f"\n- SMS to be analyzed: {sms_message}\n"
        
        if analysis.get('is_URL') and analysis.get('URLs') != "non":
            urls = analysis.get('URLs', {})
            if len(urls) > 1:
                prompt += f"- There are {len(urls)} URLs in the SMS.\n"
            
            for url_idx, url_data in urls.items():
                url = url_data.get('URL', '')
                if len(urls) > 1:
                    prompt += f"- URL {url_idx}: {url}\n"
                else:
                    prompt += f"- URL: {url}\n"
                
                # Add analysis data if available
                if url_data.get('redirect_chain') not in [None, "non"]:
                    prompt += f"- Redirect Chain of {url}: {url_data['redirect_chain']}\n"
                
                if url_data.get('html_summary') not in [None, "non"]:
                    prompt += f"- Html Content Summary of {url}: {url_data['html_summary']}\n"
                
                if url_data.get('domain_info') not in [None, "non"]:
                    prompt += f"- Domain Information of {url}: {url_data['domain_info']}\n"
                
                if url_data.get('Image_content') not in [None, "non"]:
                    prompt += f"- Screenshot Description {url}: {url_data['Image_content']}\n"
                
                if url_data.get('brand_search') not in [None, "non"] and analysis.get('is_brand'):
                    brands = analysis.get('brands', [])
                    if len(brands) > 1:
                        prompt += f"- There are {len(brands)} brands referred in the SMS.\n"
                    
                    for brand_idx, brand in enumerate(brands):
                        prompt += f"- Brand {brand_idx}: {brand}\n"
                        brand_domains = url_data.get('brand_search', {}).get(brand_idx, {}).get('brand_domain', [])
                        prompt += f"- The top five results from a Google search of the brand name: {brand_domains}\n"
        else:
            prompt += "- No URL in the SMS.\n"
        
        return prompt
    
    def _get_logits_for_text(self, text, output_dir: str = "output", enable_redirect_chain: bool = True,
                             enable_brand_search: bool = True, enable_screenshot: bool = True,
                             enable_html_content: bool = True, enable_domain_info: bool = True):
        """
        Get logits for Legitimate and Smishing tokens for a single text using full workflow.
        
        Args:
            text: Input text string
            output_dir: Directory for temporary files (screenshots, etc.)
            enable_redirect_chain: Whether to analyze URL redirect chains
            enable_brand_search: Whether to search for brand domains
            enable_screenshot: Whether to take website screenshots
            enable_html_content: Whether to analyze HTML content
            enable_domain_info: Whether to get domain information
            
        Returns:
            torch.Tensor: Logits tensor of shape [2] with [legitimate_logit, smishing_logit]
        """
        # Step 1: Extract URLs and brands from SMS
        initial_analysis = self._extract_urls_and_brands(text)
        
        # Step 2: Analyze URLs if present
        if initial_analysis.get('is_URL') and initial_analysis.get('URLs') != "non":
            urls = initial_analysis['URLs']
            if isinstance(urls, list):
                url_analysis = self._analyze_urls(
                    urls,
                    output_dir,
                    enable_redirect_chain,
                    enable_brand_search,
                    enable_screenshot,
                    enable_html_content,
                    enable_domain_info
                )
                initial_analysis['URLs'] = url_analysis
                
                # Brand search if enabled and brands detected
                if enable_brand_search and initial_analysis.get('is_brand') and initial_analysis.get('brands') != "non":
                    brands = initial_analysis['brands']
                    if isinstance(brands, list):
                        brand_analysis = self._search_brand_domains(brands)
                        for url_idx in url_analysis:
                            initial_analysis['URLs'][url_idx]['brand_search'] = brand_analysis
        
        # Step 3: Build detection prompt
        detection_prompt = self._build_detection_prompt(text, initial_analysis)
        
        # Step 4: Call Qwen with detection prompt
        try:
            detection_result = self._call_qwen_for_json(detection_prompt)
        except Exception as e:
            print(f"Detection analysis failed: {e}")
            # Fallback: use simple prompt and extract logits from tokens
            return self._get_simple_logits(text)
        
        # Step 5: Parse JSON response to extract category field
        category = detection_result.get('category', True)
        
        # Step 6: Map category to logits
        # category: True = Smishing/Phishing, False = Legitimate
        # We need to return logits, so we'll construct them based on the category
        # For TextAttack compatibility, we want higher logit for the predicted class
        
        if category is True:
            # Smishing detected - set smishing_logit high, legitimate_logit low
            smishing_logit = 10.0
            legitimate_logit = -10.0
        else:
            # Legitimate - set legitimate_logit high, smishing_logit low
            legitimate_logit = 10.0
            smishing_logit = -10.0
        
        return torch.tensor([legitimate_logit, smishing_logit])
    
    def _get_simple_logits(self, text):
        """
        Fallback method: Get logits using simple prompt and token extraction.
        Used when full workflow fails.
        """
        # Format prompt
        prompt = self._format_smishing_prompt(text)
        
        # Format using chat template (multimodal structure, text-only)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        formatted_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = self.processor.tokenizer([formatted_text], return_tensors="pt").to(self.device)
        
        # Forward pass (no generation, just get logits)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # Shape: [batch=1, seq_len, vocab_size]
        
        # Get logits at the last token position (where model would generate next token)
        last_token_logits = logits[0, -1, :]  # Shape: [vocab_size]
        
        # Extract logits for Legitimate and Smishing tokens
        legitimate_logit = last_token_logits[self.legitimate_token_id].item()
        smishing_logit = last_token_logits[self.smishing_token_id].item()
        
        return torch.tensor([legitimate_logit, smishing_logit])
    
    def __call__(self, text_input_list, **kwargs):
        """
        TextAttack-compatible method for classification.
        
        Args:
            text_input_list: List of input strings (required by TextAttack)
            **kwargs: Additional arguments (ignored for now)
            
        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, 2)
                         Each row: [legitimate_logit, smishing_logit]
        """
        if not isinstance(text_input_list, list):
            text_input_list = [text_input_list]
        
        # Process each text and collect logits
        batch_logits = []
        for text in text_input_list:
            logits = self._get_logits_for_text(text)
            batch_logits.append(logits)
        
        # Stack into batch tensor: shape (batch_size, 2)
        return torch.stack(batch_logits)
    
    def predict_proba(self, text_input_list):
        """
        Get probability predictions (optional convenience method).
        
        Args:
            text_input_list: List of input strings
            
        Returns:
            torch.Tensor: Probabilities tensor of shape (batch_size, 2)
                         Each row: [P(legitimate), P(smishing)]
        """
        logits = self.__call__(text_input_list)
        return F.softmax(logits, dim=-1)
    
    def predict(self, text_input_list):
        """
        Get class predictions (optional convenience method).
        
        Args:
            text_input_list: List of input strings
            
        Returns:
            list: List of predicted labels ("Legitimate" or "Smishing")
        """
        logits = self.__call__(text_input_list)
        predictions = torch.argmax(logits, dim=-1)
        return ["Smishing" if pred == 1 else "Legitimate" for pred in predictions]


if __name__ == "__main__":
    # Demo usage
    print("=" * 80)
    print("Qwen Smishing Classifier Demo")
    print("=" * 80)
    
    # Initialize classifier
    classifier = QwenSmishingClassifier()
    
    # Test examples
    test_texts = [
        "Your package arrives at the Cyprus Post Office tomorrow.Confirm delivery: https://51.fi/aJzP", # smishing example
        "Shipped: Your Amazon package with Old Spice High Endurance Deodorant will be delivered Tue, May 24. Track at http://a.co/4SJitSA", # legitimate example
        "FREE entry into our £250 weekly comp just send the word ENTER to 88877 NOW. 18 T&C www.textcomp.com", # spam example
    ]
    
    print("\n" + "=" * 80)
    print("Testing Classification")
    print("=" * 80)
    
    for text in test_texts:
        print(f"\nText: {text}")
        
        # Get logits (TextAttack-compatible)
        logits = classifier([text])  # Note: expects list
        print(f"Logits [Legitimate, Smishing]: {logits[0].tolist()}")
        
        # Get probabilities
        probs = classifier.predict_proba([text])
        print(f"Probabilities [P(Legitimate), P(Smishing)]: {probs[0].tolist()}")
        
        # Get prediction
        pred = classifier.predict([text])
        print(f"Prediction: {pred[0]}")
    
    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)

