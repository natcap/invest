#ifndef __LRUCACHE_H_INCLUDED__
#define __LRUCACHE_H_INCLUDED__

#include <list>
#include <map>
#include <assert.h>

using namespace std;

template <class KEY_T, class VAL_T,
    typename ListIter = typename list< pair<KEY_T,VAL_T> >::iterator,
    typename MapIter = typename map<KEY_T, ListIter>::iterator > class LRUCache{
private:
    // item_list keeps track of the order of which elements have been accessed
    // element at begin is most recent, element at end is least recent.
    // first element in the pair is its key while the second is the element
    list< pair<KEY_T,VAL_T> > item_list;
    // item_map maps an element's key to its location in the `item_list`
    // used to make lookups O(log n) time
    map<KEY_T, ListIter> item_map;
    size_t cache_size;
private:
    void clean(list< pair<KEY_T, VAL_T> > &removed_value_list){
        while(item_map.size()>cache_size){
            ListIter last_it = item_list.end(); last_it --;
            removed_value_list.push_back(
                make_pair(last_it->first, last_it->second));
            item_map.erase(last_it->first);
            item_list.pop_back();
        }
    };
public:
    LRUCache(int cache_size_):cache_size(cache_size_){
            ;
    };

    ListIter begin() {
        return item_list.begin();
    }

    ListIter end() {
        return item_list.end();
    }

    void put(
            const KEY_T &key, const VAL_T &val,
            list< pair<KEY_T, VAL_T> > &removed_value_list) {
        MapIter it = item_map.find(key);
        if(it != item_map.end()){
            // it's already in the cache, delete the location in the item
            // list and in the lookup map
            item_list.erase(it->second);
            item_map.erase(it);
        }
        // insert a new item in the front since it's most recently used
        item_list.push_front(make_pair(key,val));
        // record its iterator in the map
        item_map.insert(make_pair(key, item_list.begin()));
        // possibly remove any elements that have exceeded the cache size
        return clean(removed_value_list);
    };
    bool exist(const KEY_T &key){
        return (item_map.count(key)>0);
    };
    VAL_T& get(const KEY_T &key){
        MapIter it = item_map.find(key);
        assert(it!=item_map.end());
        // move the element to the front of the list
        item_list.splice(item_list.begin(), item_list, it->second);
        return it->second->second;
    };
};
#endif
