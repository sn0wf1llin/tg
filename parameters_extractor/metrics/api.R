# no_miss_api.R

library(dplyr)
#library(data.table)
#library(readr)
#library(rvest)
#library(XML)
library(tm)
#library(openNLP)
library(syuzhet)
#library(slam)
library(koRpus)
#library(randomForest)
#library(xgboost)  # the main algorithm
library(archdata) # for the sample dataset
#library(caret)
library(textcat)
library(quanteda)
library(corpus)
#library(tidyverse)
#library(tokenizers)
rm(list=ls())

#* @post /emotions
emotions_detect <- function(txt) {

  lang_lang <-textcat::textcat(txt)

  p <- 'en' #default value
  if (lang_lang == 'german')
  {
    p <- 'de'
  }

  if (lang_lang == 'english')
  {
    p <- 'en'
  }
  if (lang_lang != 'german' & lang_lang != 'english')
  {
    lang_lang <- 'english'
    p <- 'en'
  }

  txt_sent <- syuzhet::get_sentences(txt)

  emo_in_text_vector <- eval(parse(text=paste0('as.numeric(syuzhet::get_sentiment(as.character(txt_sent), method="nrc",language = "',lang_lang,'"))')))
  mean_emo_in_text <- mean(emo_in_text_vector) # mean
  #print(mean_emo_in_text)
  sum_emo_in_text <- sum(emo_in_text_vector) #sum

  # get count of words with different emotionality  in name
  anger_in_text <- get_nrc_sentiment(gsub("[[:punct:]]", "", txt), language = lang_lang)$anger
  anticipation_in_text <- get_nrc_sentiment(gsub("[[:punct:]]", "", txt), language = lang_lang)$anticipation
  disgust_in_text <- get_nrc_sentiment(gsub("[[:punct:]]", "", txt), language = lang_lang)$disgust
  fear_in_text <- get_nrc_sentiment(gsub("[[:punct:]]", "", txt), language = lang_lang)$fear
  joy_in_text <- get_nrc_sentiment(gsub("[[:punct:]]", "", txt), language = lang_lang)$joy
  sadness_in_text <- get_nrc_sentiment(gsub("[[:punct:]]", "", txt), language = lang_lang)$sadness
  surprise_in_text <- get_nrc_sentiment(gsub("[[:punct:]]", "", txt), language = lang_lang)$surprise
  trust_in_text <- get_nrc_sentiment(gsub("[[:punct:]]", "", txt), language = lang_lang)$trust
  negative_in_text <- get_nrc_sentiment(gsub("[[:punct:]]", "", txt), language = lang_lang)$negative
  positive_in_text <- get_nrc_sentiment(gsub("[[:punct:]]", "", txt), language = lang_lang)$positive

  emotions_text <- c(mean_emo_in_text, anger_in_text, anticipation_in_text, disgust_in_text, fear_in_text, joy_in_text,
                     sadness_in_text, surprise_in_text, trust_in_text, negative_in_text, positive_in_text)
  names(emotions_text) <- c("words_diff_emotions", "angry", "anticipation", "disgust", "fear", "joy",
                            "sadness", "surprise", "trust", "neg", "pos")
  c("words_diff_emotions", "angry", "anticipation", "disgust", "fear", "joy",
    "sadness", "surprise", "trust", "neg", "pos")
  return(emotions_text)
}



#* @post /metrics
get_add_parameters <- function(txt) {
  lang_lang <-textcat::textcat(txt)

  p <- 'en' #default value
  if (lang_lang == 'german')
  {
    p <- 'de'
  }

  if (lang_lang == 'english')
  {
    p <- 'en'
  }
  if (lang_lang != 'german' & lang_lang != 'english')
  {
    lang_lang <- 'english'
    p <- 'en'
  }
  txt <- c(txt)
  tagged.text <- eval(parse(text=paste0("koRpus::tokenize(txt, format = 'obj', lang = '",p,"')")))
  #tagged.text <- koRpus::tokenize(txt, format = 'obj', lang = 'de')

  # Daniel`s - Brawn`s Formulas
  danielson.bryan(tagged.text) -> danielson.bryan
  danielson.bryan(tagged.text)@Danielson.Bryan$DB1 -> DB1                    # !!
  danielson.bryan(tagged.text)@Danielson.Bryan$DB2 -> DB2                    # !!
  danielson.bryan(tagged.text)@Danielson.Bryan$DB2.grade.min -> DB_grade     # !!

  # Dickes-Steiwer Formulas
  dickes.steiwer(tagged.text) -> dickes.steiwer
  dickes.steiwer@Dickes.Steiwer$Dickes.Steiwer -> DS                         # !!

  # Lexical diversity
  # Herdanis C
  C.ld(tagged.text) -> Herdan                                                # !!
  Herdan@C.ld -> Herdan

  # Carroll's CTTR
  CTTR(tagged.text) -> CTTR
  CTTR@CTTR -> CTTR                                                          # !!

  # HD-D
  HDD(tagged.text) -> HDD
  HDD@HDD$HDD -> HDD                                                         # !!

  # Yule's K
  K.ld(tagged.text) -> yueles_K
  yueles_K@K.ld -> yueles_K                                                  # !!

  # Maas'
  maas(tagged.text) -> maas_1
  maas_1@Maas -> maas_1                                                      # !!

  # Moving-Average Type-Token Ratio (MATTR)
  MATTR(tagged.text) -> MATTR
  MATTR@MATTR$MATTR -> MATTR                                                 #  - !!

  # Mean Segmental Type-Token Ratio (MSTTR)
  MSTTR(tagged.text) -> MSTTR
  MSTTR@MSTTR$MSTTR -> MSTTR                                                 #  - !!

  # Measure of Textual Lexical Diversity (MTLD)
  MTLD(tagged.text) -> MTLD
  MTLD@MTLD$MTLD -> MTLD                                                     # !!

  # Guiraudis R
  R.ld(tagged.text) -> RLd
  RLd@R.ld -> RLd                                                            # !!

  # Summeris S
  S.ld(tagged.text) -> SLd
  SLd@S.ld -> SLd                                                            # !!

  # Uber
  U.ld(tagged.text) -> uber
  uber@U.ld -> uber

  # measures of relative vocabulary growth
  maas(tagged.text) -> maas
  as.numeric(maas@Maas.grw[3]) -> growth_vocabl #for example 0.25 means 25 new types every 100 tokens

  features_text <- c(DB1, DB2, DB_grade, DS, Herdan, CTTR, HDD, yueles_K, maas_1, MTLD, RLd, SLd, uber, growth_vocabl)
  names(features_text) <- c("DB1", "DB2", "DB_grade", "DS", "Herdan", "CTTR", "HDD", "yueles_K", "maas_1", "MTLD", "RLd", "SLd", "uber", "growth_vocabl")

  return(features_text)

}

#* @post /typestokens
types_tokens_methods <- function(txt) {

  txt <- c(txt)
  count_of_types_text <- summary(ntype(txt))
  as.numeric(count_of_types_text[1]) -> count_of_types_text

  count_of_tokens_text <- summary(ntoken(txt))
  as.numeric(count_of_tokens_text[1]) -> count_of_tokens_text

  ttvector <- c(count_of_tokens_text, count_of_types_text)
  names(ttvector) <- c("count_of_tokens_text", "count_of_types_text")

  return(ttvector)

}

#* @post /emotions_narrative_plot_data
narrative_emotions_plot <- function(txt) {



  txt <-gsub("\\+", " ",gsub("\\++", "+", txt))

  txt <- gsub("\n", "", txt)
  txt <- gsub("\r", "", txt)
  txt <- gsub("\t", "", txt)

  # lang_detection
  lang_lang <-textcat::textcat(txt)

  p <- 'en' #default value
  if (lang_lang == 'german')
  {
    p <- 'de'
  }

  if (lang_lang == 'english')
  {
    p <- 'en'
  }
  if (lang_lang != 'german' & lang_lang != 'english')
  {
    lang_lang <- 'english'
    p <- 'en'
  }

  txt <- gsub('J. ', 'J.', gsub('\\!([A-Z])', '\\! \\1',gsub('\\?([A-Z])', '\\? \\1', gsub('\\?([A-Z])', '\\? \\1',gsub('\\.([A-Z])', '\\. \\1', txt)))))
  txt <- gsub('Gov. ', 'Gov.', txt)

  df <- data_frame(Example_Text = txt)
  dd <-text_split(df$Example_Text, "sentences")

  df_sentences <- data_frame()
  for (k in 1:length(dd$text))
  {
    df_sentences <- rbind(df_sentences, as.data.frame(as.character(unlist(dd$text[[k]])), stringsAsFactors = F))
  }


  colnames(df_sentences) <- c('sentt')


  df_sentences <- df_sentences %>% mutate(sentt =gsub('J\\.([[:alpha:]])', 'J\\. \\1', sentt)) %>%
    mutate(sentt=gsub('Gov\\.([[:alpha:]])', 'Gov\\. \\1', sentt))

  df_sentences <- cbind(df_sentences, char_lenght =df_sentences %>%
                          apply(., 1,function(x) {nchar(x[1])}))

  df_sentences %>% mutate(end_index = cumsum(char_lenght)) -> df_sentences

  df_sentences %>% mutate(start_index = lag(end_index+1,1, default = 1))-> df_sentences


  text_s <- df_sentences$sentt
  r_number <- 1:length(text_s)

  r_number_quant <- quantile(r_number, type = 3, probs= seq(0,1,0.05))[2:21]

  r_number %>% as.data.frame() -> r_number
  colnames(r_number) <- c('row_n')

  text_s <-cbind(text_s,r_col = r_number %>% apply(.,1, function(x)
  {

    r_number_quant[min(which(x[1]<=r_number_quant))]

  }))

  text_s <- data.frame(text_s, stringsAsFactors = F)

  text_s$r_col <- as.integer(text_s$r_col)

  df_sentences$used_id <- text_s$r_col

  text_s %>%
    group_by(r_col) %>%
    summarise( sent = mean(syuzhet::get_sentiment(as.character(text_s), method="nrc",language = lang_lang))) %>%
    ungroup() %>%
    mutate(rn = row_number()) -> text_s


  norm_value <- max(abs(text_s$sent))

  text_s <- text_s %>% mutate(sent_n = round(sent/norm_value,1))

  df_sentences %>% group_by(used_id) %>% summarise(start_index = min(start_index),
                                                   end_index = max(end_index)) %>% ungroup() %>%
    select(start_index, end_index) -> index_df


  index_df


  narrative_emotions_data <- list(x_narrative=text_s$rn, y_emotions=text_s$sent_n, startInd = index_df$start_index, endInd = index_df$end_index)

  return(narrative_emotions_data)

}


#* @post /highlights

highlights <- function(txt) {
  txt <- paste0(txt," ")
  text <-gsub("\\+", " ",gsub("\\++", "+", txt))

  text <- gsub("\n", "", text)
  text <- gsub("\r", "", text)
  text <- gsub("\t", "", text)

  # lang_detection
  lang_lang <-textcat::textcat(txt)

  p <- 'en' #default value
  if (lang_lang == 'german')
  {
    p <- 'de'
  }

  if (lang_lang == 'english')
  {
    p <- 'en'
  }
  if (lang_lang != 'german' & lang_lang != 'english')
  {
    lang_lang <- 'english'
    p <- 'en'
  }

  #text <- gsub('J. ', 'J.', gsub('\\!([[A-Z]])', '\\! \\1',gsub('\\?([[A-Z]])', '\\? \\1', gsub('\\?([[A-Z]])', '\\? \\1',gsub('\\.([[A-Z]])', '\\. \\1', text)))))
  text <- gsub('J. ', 'J.', gsub('\\!([A-Z])', '\\! \\1',gsub('\\?([A-Z])', '\\? \\1', gsub('\\?([A-Z])', '\\? \\1',gsub('\\.([A-Z])', '\\. \\1', text)))))
  # Gov.
  text <- gsub('Gov. ', 'Gov.', text)

  #text <- gsub('J. ', 'J.', gsub('\\!([[:alpha:]])', '\\! \\1',gsub('\\?([[:alpha:]])', '\\? \\1', gsub('\\?([[:alpha:]])', '\\? \\1',gsub('\\.([[:alpha:]])', '\\. \\1', text)))))

  df <- data_frame(Example_Text = text)
  dd <-text_split(df$Example_Text, "sentences")

  df_sentences <- data_frame()
  for (k in 1:length(dd$text))
  {
    df_sentences <- rbind(df_sentences, as.data.frame(as.character(unlist(dd$text[[k]])), stringsAsFactors = F))
  }


  colnames(df_sentences) <- c('sentt')


  df_sentences <- df_sentences %>% mutate(sentt =gsub('J\\.([[:alpha:]])', 'J\\. \\1', sentt)) %>%
    mutate(sentt=gsub('Gov\\.([[:alpha:]])', 'Gov\\. \\1', sentt))

  df_sentences <- cbind(df_sentences, char_lenght =df_sentences %>%
                          apply(., 1,function(x) {nchar(x[1])}))

  df_sentences %>% mutate(end_index = cumsum(char_lenght)) -> df_sentences

  df_sentences %>% mutate(start_index = lag(end_index+1,1, default = 1))-> df_sentences


  avg_sentences_lenght <- mean(sapply(tokenizers::tokenize_words(df_sentences[[1]]), length))

  sentences_lenght <- sapply(tokenizers::tokenize_words(df_sentences[[1]]), length)

  df_sentences$l_words <- sentences_lenght

  df_sentences %>% mutate(is_short = round(l_words/avg_sentences_lenght,1),
                          is_long = round(l_words/avg_sentences_lenght,1)) -> df_sentences


  top_short <- df_sentences %>% filter(is_short <= 0.5)
  top_long <- df_sentences %>% filter(is_long >= 2)


  res1 <- character()

  if (nrow(top_short)>0) {
    for (j in 1:nrow(top_short))
    {
      start_index <- top_short$start_index[j]
      end_index <- top_short$end_index[j]
      res1 <- paste0(res1, "{\"startIndex\": ",start_index-1,", ","\"endIndex\": ", end_index-1,", ", " \"rule\": ", "\"too_short\"},")
    }
  }

  res2 <- character()
  if (nrow(top_long)>0) {
    for (j in 1:nrow(top_long))
    {

      start_index <- top_long$start_index[j]
      end_index <- top_long$end_index[j]

      res2 <- paste0(res2, "{\"startIndex\": ",start_index-1,", ","\"endIndex\": ", end_index-1,", ", "\"rule\": ", "\"too_long\"},")
    }
  }

  s_v <- df_sentences$sentt

  s_v_sentiment <- eval(parse(text=paste0('as.numeric(syuzhet::get_sentiment(as.character(s_v), method="syuzhet",language = "',lang_lang,'"))')))

  df_sentences$emotion <- s_v_sentiment


  top_negative <- df_sentences %>% filter(emotion < -2)
  top_positive <- df_sentences %>% filter(emotion > 2)


  res3 <- character()

  if (nrow(top_negative)>0) {
    for (j in 1:nrow(top_negative))
    {
      start_index <- top_negative$start_index[j]
      end_index <- top_negative$end_index[j]
      res3 <- paste0(res3, "{\"startIndex\": ",start_index-1,", ","\"endIndex\": ", end_index-1,", ", "\"rule\": ", "\"too_negative\"},")
    }
  }

  res4 <- character()

  if (nrow(top_positive)>0) {
    for (j in 1:nrow(top_positive))
    {
      start_index <- top_positive$start_index[j]
      end_index <- top_positive$end_index[j]
      res4 <- paste0(res4, "{\"startIndex\": ",start_index-1,", ","\"endIndex\": ", end_index-1,", ", "\"rule\": ", "\"too_positive\"},")
    }
  }

  res <- paste0("[",res1, res2,res3, res4,"]")


  res <- gsub("},]", "}]", res)
  substr(res, max(gregexpr("},", res)[[1]]), max(gregexpr("},", res)[[1]])+2) <- "}"


  return (res)
}

#* @post /refactored_original_text
refactored_original_text <- function(txt) {
  txt <- paste0(txt," ")
  text <-gsub("\\+", " ",gsub("\\++", "+", txt))

  text <- gsub("\n", "", text)
  text <- gsub("\r", "", text)
  text <- gsub("\t", "", text)
  text <- gsub('J. ', 'J.', gsub('\\!([A-Z])', '\\! \\1',gsub('\\?([A-Z])', '\\? \\1', gsub('\\?([A-Z])', '\\? \\1',gsub('\\.([A-Z])', '\\. \\1', text)))))
  text <- gsub('Gov. ', 'Gov.', text)

  #text <- gsub('J. ', 'J.', gsub('\\!([[:alpha:]])', '\\! \\1',gsub('\\?([[:alpha:]])', '\\? \\1', gsub('\\?([[:alpha:]])', '\\? \\1',gsub('\\.([[:alpha:]])', '\\. \\1', text)))))

  df <- data_frame(Example_Text = text)
  dd <-text_split(df$Example_Text, "sentences")

  df_sentences <- data_frame()
  for (k in 1:length(dd$text))
  {
    df_sentences <- rbind(df_sentences, as.data.frame(as.character(unlist(dd$text[[k]])), stringsAsFactors = F))
  }


  colnames(df_sentences) <- c('sentt')


  df_sentences <- df_sentences %>% mutate(sentt =gsub('J\\.([[:alpha:]])', 'J\\. \\1', sentt)) %>%
    mutate(sentt=gsub('Gov\\.([[:alpha:]])', 'Gov\\. \\1', sentt))


  df_sentences <- cbind(df_sentences, char_lenght =df_sentences %>%
                          apply(., 1,function(x) {nchar(x[1])}))

  df_sentences %>% mutate(end_index = cumsum(char_lenght)) -> df_sentences

  df_sentences %>% mutate(start_index = lag(end_index + 1,1, default = 1))-> df_sentences



  res5 <- '{"text":"'

  if (nrow(df_sentences)>0) {
    for (j in 1:nrow(df_sentences))
    {

      res5 <- paste0(res5,paste0(df_sentences$sentt[j], ""))
    }
  }
  res5 <- paste0(res5, '"}')

  res <- paste0("[", res5,"]")


  res <- gsub("},]", "}]", res)



  return (res)
}

