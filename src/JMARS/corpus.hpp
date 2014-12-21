#pragma once
//Copyright [2014] [Wei Zhang]

//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//http://www.apache.org/licenses/LICENSE-2.0
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

///////////////////////////////////////////////////////////////////
// Date: 2014/5/14                                               //
// Class for review data management.                             //
///////////////////////////////////////////////////////////////////

#include "../utils.hpp"


/// To sort words by frequency in a corpus
bool wordCountCompare(std::pair<std::string, int> p1, std::pair<std::string, int> p2) {
    return p1.second > p2.second;
}

/// To sort votes by product ID
bool voteCompare(Vote* v1, Vote* v2) {
    return v1->item > v2->item;
}

/// Sign (-1, 0, or 1)
template<typename T> int sgn(T val) {
    return (val > T(0)) - (val < T(0));
}

class Corpus {
public:
    Corpus(char* trdata_path, char* vadata_path,
            char* tedata_path, int max_words, bool pos_tag) {
        std::map<std::string, int> u_counts;
        std::map<std::string, int> i_counts;
        std::string u_name;
        std::string b_name;
        float value;
        std::string vote_time;
        int nw;
        int n_read = 0;

        // Read the input file. The first time the file is read it is only to
        // compute word counts, in order to select the top "maxWords" words to
        // include in the dictionary
        std::ifstream* in = utils::ifstream_(trdata_path);
        std::string line, s_word, pos_word;
        std::vector<std::string> parts;
        while (std::getline(*in, line)) {
            std::stringstream ss(line);
            ss >> u_name >> b_name >> value >> vote_time >> nw;
            if (value > 5 or value < 0) { 
                // Ratings should be in the range [0,5]
                printf("Got bad value of %f\nOther fields were %s %s %s\n", 
                        value, u_name.c_str(), b_name.c_str(), vote_time.c_str());
                exit(0);
            }
            for (int w = 0; w < nw; w++) {
                if (pos_tag) {
                    ss >> pos_word;
                    parts = utils::split_str(pos_word, '_');
                    s_word = parts[0];
                } else {
                    ss >> s_word;
                }
                if (word_count.find(s_word) == word_count.end())
                    word_count[s_word] = 0;
                word_count[s_word]++;
                parts.clear();
                std::vector<std::string>(parts).swap(parts);
            }
            
            // Count number of revivews each user and item has
            // (As I cleaned data previously, this procedure may not function.)
            if (u_counts.find(u_name) == u_counts.end())
                u_counts[u_name] = 0;
            if (i_counts.find(b_name) == i_counts.end())
                i_counts[b_name] = 0;
            u_counts[u_name]++;
            i_counts[b_name]++;

            n_read++;
            if (n_read % 100000 == 0) {
                printf(".");
                fflush(stdout);
            }
        }
        in->close();

        printf("\nBasic statisticals: nUsers = %d, nItems = %d, nRatings = %d\n",
                (int)u_counts.size(), (int)i_counts.size(), n_read);

        // Sorting words according to their frequency, the save only top 
        //   "max_words", specified by user
        std::vector < std::pair<std::string, int> > which_words;
        for (std::map<std::string, int>::iterator it = word_count.begin();
                it != word_count.end(); it++)
            which_words.push_back(*it);
        sort(which_words.begin(), which_words.end(), wordCountCompare);
        if ((int) which_words.size() < max_words)
            max_words = (int) which_words.size();
        n_words = max_words;
        for (int w = 0; w < max_words; w++) {
                word_ids[which_words[w].first] = w;
                rword_ids[w] = which_words[w].first;
        }
    
        // Loading rating and review text data in memory
        n_users = 0;
        n_items = 0;
        n_pos = 0;
        TR_V = utils::loadReviewData(trdata_path,
                                     &word_ids,
                                     &user_ids,
                                     &ruser_ids,
                                     &item_ids,
                                     &ritem_ids,
                                     &pos_ids,
                                     &rpos_ids,
                                     n_users,
                                     n_items,
                                     n_pos);
        VA_V = utils::loadReviewData(vadata_path,
                                     &word_ids,
                                     &user_ids,
                                     &ruser_ids,
                                     &item_ids,
                                     &ritem_ids,
                                     &pos_ids,
                                     &rpos_ids,
                                     n_users,
                                     n_items,
                                     n_pos);
        TE_V = utils::loadReviewData(tedata_path,
                                     &word_ids,
                                     &user_ids,
                                     &ruser_ids,
                                     &item_ids,
                                     &ritem_ids,
                                     &pos_ids,
                                     &rpos_ids,
                                     n_users,
                                     n_items,
                                     n_pos);
    }

    ~Corpus() {
        for (std::vector<Vote*>::iterator it = TR_V->begin();
                it != TR_V->end(); it++)
            delete *it;
        delete TR_V;
        for (std::vector<Vote*>::iterator it = VA_V->begin();
                it != VA_V->end(); it++)
            delete *it;
        delete VA_V;
        for (std::vector<Vote*>::iterator it = TE_V->begin();
                it != TE_V->end(); it++)
            delete *it;
        delete TE_V;
        user_ids.clear();
        std::map<std::string, int>(user_ids).swap(user_ids);
        ruser_ids.clear();
        std::map<int, std::string>(ruser_ids).swap(ruser_ids);
        item_ids.clear();
        std::map<std::string, int>(item_ids).swap(item_ids);
        ritem_ids.clear();
        std::map<int, std::string>(ritem_ids).swap(ritem_ids);
        word_ids.clear();
        std::map<std::string, int>(word_ids).swap(word_ids);
        rword_ids.clear();
        std::map<int, std::string>(rword_ids).swap(rword_ids);
        pos_ids.clear();
        std::map<std::string, int>(pos_ids).swap(pos_ids);
        rpos_ids.clear();
        std::map<int, std::string>(rpos_ids).swap(rpos_ids);
    }

    std::vector<Vote*>* TR_V;     // storing training data
    std::vector<Vote*>* TE_V;     // storing testing data
    std::vector<Vote*>* VA_V;     // storing validation data

    int n_users; // Number of users
    int n_items; // Number of items
    int n_words; // Number of words
    int n_pos;   // Number of pos-tagging

    std::map<std::string, int> user_ids; // Map a user's string ID to integer
    std::map<std::string, int> item_ids; // Map an item's string ID to integer

    std::map<int, std::string> ruser_ids; // Inverse of the above map
    std::map<int, std::string> ritem_ids;

    std::map<std::string, int> word_count; // Frequency of each word in the corpus
    std::map<std::string, int> word_ids; // Map each word to its integer ID
    std::map<int, std::string> rword_ids; // Inverse of the above map

    std::map<std::string, int> pos_ids; // Map pos tagging to integer ID
    std::map<int, std::string> rpos_ids;
};
