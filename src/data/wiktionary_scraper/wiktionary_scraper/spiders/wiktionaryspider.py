import scrapy


class WiktionarySpider(scrapy.Spider):
    name = 'wiktionary'
    start_urls = ['https://en.wiktionary.org/w/index.php?title=Category:Polish_feminine_nouns&from=A',
                  'https://en.wiktionary.org/w/index.php?title=Category:Polish_masculine_nouns&from=A',
                  'https://en.wiktionary.org/w/index.php?title=Category:Polish_neuter_nouns&from=A',
                  'https://en.wiktionary.org/w/index.php?title=Category:German_masculine_nouns&from=A',
                  'https://en.wiktionary.org/w/index.php?title=Category:German_feminine_nouns&from=A',
                  'https://en.wiktionary.org/w/index.php?title=Category:German_neuter_nouns&from=A',
                  'https://en.wiktionary.org/w/index.php?title=Category:French_feminine_nouns&from=A',
                  'https://en.wiktionary.org/w/index.php?title=Category:French_masculine_nouns&from=A',
                  'https://en.wiktionary.org/w/index.php?title=Category:Spanish_feminine_nouns&from=A',
                  'https://en.wiktionary.org/w/index.php?title=Category:Spanish_masculine_nouns&from=A']

    excluded_urls = ['https://en.wiktionary.org/wiki/Category:Polish_feminine_nouns',
                     'https://en.wiktionary.org/wiki/Category:Polish_masculine_nouns',
                     'https://en.wiktionary.org/wiki/Category:Polish_neuter_nouns'
                     'https://en.wiktionary.org/wiki/Category:German_neuter_nouns',
                     'https://en.wiktionary.org/wiki/Category:German_masculine_nouns',
                     'https://en.wiktionary.org/wiki/Category:German_feminine_nouns',
                     'https://en.wiktionary.org/wiki/Category:Spanish_masculine_nouns',
                     'https://en.wiktionary.org/wiki/Category:Spanish_feminine_nouns',
                     'https://en.wiktionary.org/wiki/Category:French_feminine_nouns',
                     'https://en.wiktionary.org/wiki/Category:French_masculine_nouns'
                     ]
    visited_urls = start_urls + excluded_urls

    def parse(self, response):
        info = response.xpath("//span[@class='mw-page-title-main']/text()").get().split()
        lang = info[0]
        gender = info[1]

        # for every noun found in 'response', yield a dict with gender, lang, and noun
        for noun in response.xpath("//div[@class='mw-category-group']/ul/li/a/text()").getall():
            yield {
                'noun': noun,
                'gender': gender,
                'lang': lang,
                }
        # get all link pages (e.g 'next' page)
        pages =  response.xpath("//table[@id='toc']/tbody//a/@href").getall()
        if pages: # if list is not empty
            unvisited = [url for url in pages if url not in self.visited_urls] # filter out urls that have already been visited
            if unvisited: # if we have unvisited urls
                next_page = unvisited[0] # next_page is set to first item in unvisted
                self.visited_urls.append(next_page) # add next_page to visited list to not visit it again
                yield response.follow(next_page, callback=self.parse)