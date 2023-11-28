from googletrans import Translator
from mimix.interact import run_interactive

class ChatTranslator:
    def __init__(self):
        self.translator = Translator()

    def translate_text(self, text, target_lang):
        return self.translator.translate(text, dest=target_lang).text

    def detect_language(self, text):
        return self.translator.detect(text).lang.lower()

    def process_query(self, text):
        original_lang = self.detect_language(text)
        
        # 翻译到中文
        query_in_chinese = self.translate_text(text, 'zh-cn')

        # 调用医疗诊断模型
        diagnosis_results = run_interactive(query_in_chinese)

        # 将每个诊断结果翻译回原始语言
        translated_results = [self.translate_text(result, original_lang) for result in diagnosis_results]

        return translated_results

# Example usage
if __name__ == '__main__':
    chat_translator = ChatTranslator()

    # 用户输入测试
    user_input = input("Enter your query: ")
    translated_results = chat_translator.process_query(user_input)
    for result in translated_results:
        print(result)
