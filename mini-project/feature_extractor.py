"""
Enhanced URL Feature Extractor
Extracts 44 features from URLs (30 original + 14 red flags)
"""

import re
from urllib.parse import urlparse


class EnhancedURLFeatureExtractor:
    """Extract 44 features from raw URLs (30 original + 14 red flags)"""
    
    KNOWN_BRANDS = [
        'paypal', 'amazon', 'facebook', 'google', 'microsoft', 'apple',
        'netflix', 'ebay', 'linkedin', 'instagram', 'twitter', 'bank',
        'chase', 'wellsfargo', 'citibank', 'hsbc', 'stripe', 'visa',
        'mastercard', 'americanexpress', 'discover', 'dropbox', 'adobe',
        'yahoo', 'aol', 'gmail', 'outlook', 'icloud'
    ]
    
    SUSPICIOUS_TLDS = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', 
                       '.work', '.click', '.link', '.loan', '.download']
    
    SHORTENERS = ['bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co', 
                  'is.gd', 'buff.ly', 'adf.ly']
    
    @staticmethod
    def extract_features(url):
        """Extract all 44 features from a URL"""
        features = {}
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower() if parsed.netloc else ''
            path = parsed.path.lower()
            query = parsed.query.lower()
            scheme = parsed.scheme.lower()
            
            # ============ 30 ORIGINAL FEATURES ============
            
            # 1. having_IP_Address
            ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
            features['having_IP_Address'] = -1 if ip_pattern.match(domain) else 1
            
            # 2. URL_Length
            length = len(url)
            if length < 54:
                features['URL_Length'] = 1
            elif 54 <= length <= 75:
                features['URL_Length'] = 0
            else:
                features['URL_Length'] = -1
            
            # 3. Shortining_Service
            features['Shortining_Service'] = -1 if any(s in domain for s in EnhancedURLFeatureExtractor.SHORTENERS) else 1
            
            # 4. having_At_Symbol
            features['having_At_Symbol'] = -1 if '@' in url else 1
            
            # 5. double_slash_redirecting
            features['double_slash_redirecting'] = -1 if url.count('//') > 1 else 1
            
            # 6. Prefix_Suffix
            features['Prefix_Suffix'] = -1 if '-' in domain else 1
            
            # 7. having_Sub_Domain
            dots = domain.count('.')
            if dots == 1:
                features['having_Sub_Domain'] = 1
            elif dots == 2:
                features['having_Sub_Domain'] = 0
            else:
                features['having_Sub_Domain'] = -1
            
            # 8. SSLfinal_State
            features['SSLfinal_State'] = 1 if scheme == 'https' else -1
            
            # 9-30: Remaining features
            features['Domain_registeration_length'] = 1
            features['Favicon'] = 1
            features['port'] = -1 if parsed.port and parsed.port not in [80, 443] else 1
            features['HTTPS_token'] = -1 if 'https' in domain else 1
            features['Request_URL'] = 1
            features['URL_of_Anchor'] = 1
            features['Links_in_tags'] = 1
            features['SFH'] = 1
            features['Submitting_to_email'] = 1
            features['Abnormal_URL'] = 1
            features['Redirect'] = 0
            features['on_mouseover'] = 1
            features['RightClick'] = 1
            features['popUpWidnow'] = 1
            features['Iframe'] = 1
            features['age_of_domain'] = 1
            features['DNSRecord'] = 1
            features['web_traffic'] = 0
            features['Page_Rank'] = 0
            features['Google_Index'] = 1
            features['Links_pointing_to_page'] = 0
            features['Statistical_report'] = 1
            
            # ============ 14 RED FLAG FEATURES ============
            
            # 1. @ symbol (credential phishing)
            features['red_flag_at_symbol'] = 1 if '@' in url else -1
            
            # 2. IP address
            features['red_flag_ip_address'] = 1 if ip_pattern.match(domain) else -1
            
            # 3. HTTP on sensitive site
            sensitive_keywords = ['paypal', 'bank', 'login', 'signin', 'account', 
                                'payment', 'secure', 'verify', 'update', 'confirm']
            is_sensitive = any(kw in url.lower() for kw in sensitive_keywords)
            features['red_flag_http_sensitive'] = 1 if (scheme == 'http' and is_sensitive) else -1
            
            # 4. Brand spoofing (brand in path but not domain)
            brand_in_path = any(brand in path for brand in EnhancedURLFeatureExtractor.KNOWN_BRANDS)
            brand_in_domain = any(brand in domain for brand in EnhancedURLFeatureExtractor.KNOWN_BRANDS)
            features['red_flag_brand_spoofing'] = 1 if (brand_in_path and not brand_in_domain) else -1
            
            # 5. Excessive subdomains
            features['red_flag_excessive_subdomains'] = 1 if dots > 3 else -1
            
            # 6. Suspicious TLD
            features['red_flag_suspicious_tld'] = 1 if any(tld in domain for tld in EnhancedURLFeatureExtractor.SUSPICIOUS_TLDS) else -1
            
            # 7. Gibberish domain (low vowel ratio)
            domain_main = domain.split('.')[0] if '.' in domain else domain
            if len(domain_main) > 6:
                vowels = sum(1 for c in domain_main if c in 'aeiou')
                vowel_ratio = vowels / len(domain_main)
                features['red_flag_gibberish_domain'] = 1 if vowel_ratio < 0.2 else -1
            else:
                features['red_flag_gibberish_domain'] = -1
            
            # 8. URL shortener
            features['red_flag_url_shortener'] = 1 if any(short in domain for short in EnhancedURLFeatureExtractor.SHORTENERS) else -1
            
            # 9. Double slash in path
            features['red_flag_double_slash_path'] = 1 if '//' in path else -1
            
            # 10. Dash with brand name
            dash_with_brand = '-' in domain and any(brand in domain for brand in EnhancedURLFeatureExtractor.KNOWN_BRANDS)
            features['red_flag_dash_brand'] = 1 if dash_with_brand else -1
            
            # 11. Excessive URL length
            features['red_flag_excessive_length'] = 1 if len(url) > 100 else -1
            
            # 12. Suspicious file extension
            suspicious_ext = ['.exe', '.zip', '.scr', '.bat', '.cmd', '.apk']
            features['red_flag_suspicious_extension'] = 1 if any(ext in path for ext in suspicious_ext) else -1
            
            # 13. Non-standard port
            features['red_flag_non_standard_port'] = 1 if (parsed.port and parsed.port not in [80, 443]) else -1
            
            # 14. Data URI or JavaScript
            features['red_flag_data_uri'] = 1 if (url.startswith('data:') or 'javascript:' in url.lower()) else -1
            
        except Exception as e:
            print(f"Error extracting features from URL: {e}")
            # Return default safe values
            features = {f'feature_{i}': 1 for i in range(30)}
            for i in range(14):
                features[f'red_flag_{i}'] = -1
        
        return features
    
    @staticmethod
    def get_red_flag_explanation(features):
        """Get explanations for triggered red flags"""
        red_flags = []
        
        if features.get('red_flag_at_symbol', -1) == 1:
            red_flags.append("Contains @ symbol (credential phishing)")
        if features.get('red_flag_ip_address', -1) == 1:
            red_flags.append("Uses IP address instead of domain")
        if features.get('red_flag_http_sensitive', -1) == 1:
            red_flags.append("Uses HTTP for sensitive site (should be HTTPS)")
        if features.get('red_flag_brand_spoofing', -1) == 1:
            red_flags.append("Brand name in path but not domain (spoofing)")
        if features.get('red_flag_excessive_subdomains', -1) == 1:
            red_flags.append("Excessive subdomains (>3 levels)")
        if features.get('red_flag_suspicious_tld', -1) == 1:
            red_flags.append("Suspicious/free TLD commonly used in phishing")
        if features.get('red_flag_gibberish_domain', -1) == 1:
            red_flags.append("Gibberish domain name (low vowel ratio)")
        if features.get('red_flag_url_shortener', -1) == 1:
            red_flags.append("URL shortener (hides destination)")
        if features.get('red_flag_double_slash_path', -1) == 1:
            red_flags.append("Double slash redirect technique")
        if features.get('red_flag_dash_brand', -1) == 1:
            red_flags.append("Dash with brand name (spoofing)")
        if features.get('red_flag_excessive_length', -1) == 1:
            red_flags.append("Excessive URL length (obfuscation)")
        if features.get('red_flag_suspicious_extension', -1) == 1:
            red_flags.append("Suspicious file extension")
        if features.get('red_flag_non_standard_port', -1) == 1:
            red_flags.append("Non-standard port number")
        if features.get('red_flag_data_uri', -1) == 1:
            red_flags.append("Data URI or JavaScript (XSS risk)")
        
        return red_flags

