# This script contains some simple exploratory analysis on the original file
# to detect possible errors or problems that we need to take into account 
# when using it in our model.

library(jsonlite)
library(stringr)
library(slam)
library(tm)
library(tidyverse)

# Preprocessing -----------------------------------------------------------

reviews <- fromJSON('data/reviews.json.gz')

# Example of columns found in each restaurant reviews set
reviews$eatery_8071044$reviews %>% colnames
# "user_location"  "review_text"    "uid"            "user_badges"    "review_id"     
# "review_answers" "review_title"   "review_rating" 

# Drop any rows with NULL in nreviews (no reviews for that restaurant)
reviews <- reviews[map(reviews, ~!is.null(.$nreviews)) %>% unlist]

# How many reviews and restaurants are there?
length(reviews) # 7343 restaurants
reviews %>% map(~.$nreviews) %>% combine %>% sum # 861591 reviews

locations <- reviews %>% map(~.$reviews$user_location)
# Some review do not have user_location, so R interprets them as logical
# Filter them first, then combine all locations and show the most common ones
# Filter also blank locations (blank strings)
locations[map(locations, class) != 'logical'] %>% 
  combine %>% 
  '['(., . != '') %>% 
  table %>% sort(decreasing=TRUE) %>% 
  tibble(city=as.character(names(.)), reviews=as.numeric(.)) %>% select(-.) %>% 
  mutate(city=reorder(city, reviews)) %>% 
  head(10) %>% 
  ggplot(aes(city, reviews)) +
  geom_bar(stat='identity') + coord_flip() + 
  ggtitle('Most common user locations') +
  theme(plot.title = element_text(hjust = 0.5))

ggsave('plots/user_locations.png')

# As we can see, there are quite a few reviews from abroad, 
# which means we should expect foreign languages, not only Spanish or Catalan.

# We're interested in English reviews due to the the high availability of NLP resources
# for the English languages, and there are supposedly quite a few of them.

# Let's see some reviews
set.seed(123)
reviews %>% 
  map(~.$reviews$review_text) %>% 
  combine %>% sample(10) %>% print
# [1] "restaurant was recommended by a co-worker and it was really good. We just asked server to bring whatever he thought was fresh and good...it was soo good! There was a queue and you have to wait until people finish before you get a seat. Its a small place and easy to miss if you don't realize the restaurant is behind a garage door if you want to go early before they open. The queue gives it away though. Worth the visit to Cal Pep!"                                                                                                                                                                                                                                             
# [2] "Der Burger und die Pommes (welche selbstgemacht aussahen) waren der Hammer! :)\nIch fand das Konzept einer Speisekarte zum selbst-ausfüllen super, man merkt da hat sich einer Gedanken gemacht wie man internationale Kundschaft gut bedient.\nMit viel Liebe geführter Laden, vom Essen bis zum Service."                                                                                                                                                                                                                                                                                                                                                                                   
# [3] ""                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
# [4] "Un italiano 100% auténtico con pizzas muy buenas y pasta artesanal excelente (yo comí unos raviolis de cacao y ceps). Se traen directamente desde Italia los ingredientes que no encuentran aquí. De hecho lo hacen casi todo ellos para tener el nivel de calidad deseado, por ejemplo el tiramisù o el limoncello del chupito final.\nAdemás el servicio es excelente, son muy atentos y simpáticos. Nosotros ibamos con niños y llevábamos el carrito, una bici y un monopatín y no fue ningún problema."                                                                                                                                                                                  
# [5] "Our first day in Barcelona and we were looking for an evening meal, we had been up since 4am and are Brits so by 6.30pm we were hungry and stumbled across Mussol. The menu was reasonably priced and there were plenty of options so we wandered in.\n\nThe waiter threw the menus down on a table in the back and left us to peruse. 20 minutes later and no one had been over, four staff were chatting and not one acknowledged are closed menus.\n\nWe walked out and told a bartender why were leavin but she did nothing either.\n\nNot a great first impression of Barcelona."                                                                                                        
# [6] "Restaurante con 5 años de experiencia. Cocina muy elaborada y con gustos muy atractivos. Ofrecen dos tipos de menú y también carta."                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
# [7] "Cena con amigos, con un pica pica bueno y unos pescados muy bien elaborados, que terminanos con unos postres caseros...\nMore "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
# [8] ""                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
# [9] "Cuando era pequeña íbamos siempre que nos apetecía arroz de calidad, tenía ese recuerdo de calidad, ahora se ha convertido en un restaurante de calidad media enfocado a los turistas y tremendamente caro, para lo que dan, fuimos unas parejas, pedimos varios tipos de arroz y de tres tipos solo uno estaba bien, el resto regular, los entrantes del montón, eso sí, lo peor fue que a la hora de pagar nos cobran más de 2€ por persona por servicio de pan que no pedimos , señores la normativa catalana de consum establece que està prohibido cobrar ni servicio ni nada que el cliente no pide, en concreto pone de ejemplo el pan entre otros, con lo que espero que rectifiquen."
# [10] "Terrible. Eran las 18h, iba con mi madre y con mi tía. Había muchos turistas sentados en las mesas, tomando cervezas y sangría. Nos sentamos en la única mesa vacía que había y el que decía ser el dueño vino corriendo diciendo que si no íbamos a cenar, no nos podíamos sentar. Cuando le dije que había otros turistas tomando algo, se puso a reir, nos contestó que hacía lo que le daba la gana y que no nos servirían, y que los turistas se dejan más dinero. Que solamente cenas. Le solicité el libro de reclamaciones y me dijo que no tenía y que llamara a la urbana, que se reirían de mí. Evidentemente, nos fuimos."

# As we see here, there is English, German and Spanish in this little sample.
# If we looked at more reviews, we would also find French, Italian, Chinese, Dutch, etc.

# Notice also how review [7] ends with \nMore. Are there many reviews like that?
reviews %>% 
  map(~.$reviews$review_text) %>% 
  combine %>% 
  str_detect('\\nMore\\s*$') %>% sum # 13563

# Quite a lot, actually. Those reviews should be omitted from the dataset when preprocessing it.

# Which are the most common ratings?
reviews %>% 
  map(~.$reviews$review_rating %>% str_sub(1, 1) %>% as.integer) %>% 
  combine %>% table %>% barplot(main='Number of reviews per rating')

# There are some other fields, like uid, which contains 
# the user_id of the user who wrote the review, but we're just interested 
# in the text of the reviews, so we don't need to look at it.

# Finally, let's check that there aren't any issues with duplicates.
# Notice that there are some duplicates review_id
duplicated(
  reviews %>% 
    map(~.$reviews$review_id) %>% 
    combine()
) %>% sum # 1368

# So we need to deduplicate to avoid repeated samples in our model.