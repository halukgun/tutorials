#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
class movies{
private:
string bos;
vector <string>movieList;
public:
    void addMovie(){
    cout<<"please enter the movie name: ";
    cin>>bos;
    movieList.push_back(bos);
    bos.clear();
    cout<<"please enter the producer: ";
    cin>>bos;
    movieList.push_back(bos);
    bos.clear();
    }
    void printList(){
        for(int i=0;i<movieList.size();i=i+2){
            cout<<"movie name: "<<movieList[i]<<" producer name: "<<movieList[i+1]<<endl;
        }
    }
    vo

};

int main(){
movies a;
a.addMovie();
a.printList();
    
return 0;
}