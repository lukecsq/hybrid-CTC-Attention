#include <fstream>
#include <iostream>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <codecvt>
#include <locale>
#include <ctime>
using namespace std;

class splitorForCh
{
public:
	splitorForCh();
	void clear();
	void splitTxtCh(int n,int mxlen, string inputpath, string outputpath);
private:
	bool isCh(int x);
	void fun(wstring &wd);
	void split(wstring &ws);
	void read(string filepath);
    int limit;
	map<wstring,int> ma;
	vector<int> num;
	vector<pair<int,wstring> > ans;
};

bool cmp(pair<int, wstring> a, pair<int, wstring> b)
{
	if(a.first!=b.first) return a.first>b.first;
	if(a.second.size()!=b.second.size()) return a.second.size()<b.second.size();
	return a.second<b.second;
}

splitorForCh::splitorForCh()
{
	clear();
}

void splitorForCh::clear()
{
	num.clear();
	num.resize(100011);
	ma.clear();
	ans.clear();
}

void splitorForCh::splitTxtCh(int n, int mxlen, string inputpath, string outputpath)
{
	limit = mxlen;
	read(inputpath);
	for(map<wstring,int>::iterator it = ma.begin();it!=ma.end();it++)
		if(it->second>100000) num[100001]++;
		else num[it->second]++;
	int minum = 0;
	long long sum = 0;
	for(int i=100001;i>=1;i--)
	{
		sum+=num[i];
		if(sum>=n)
		{
			minum = i;
			break;
		}
	}
	ofstream fout("frequency+"+outputpath, ios::out);
    ofstream fout2(outputpath, ios::out);
	for(map<wstring,int>::iterator it = ma.begin();it!=ma.end();it++)
	{
		if(it->second>minum)
		{
			n--;
			ans.push_back(make_pair(it->second,it->first));
		}
	}
	for(map<wstring,int>::iterator it = ma.begin();it!=ma.end();it++)
	{
		if(it->second==minum)
		{
			n--;
			ans.push_back(make_pair(it->second,it->first));
			if(!n) break;
		}
	}
	sort(ans.begin(),ans.end(),cmp);
	wstring_convert<codecvt_utf8<wchar_t>> conv;
	for(int i=0;i<(int)ans.size();i++)
		fout<<conv.to_bytes(ans[i].second)<<" "<<ans[i].first<<endl;
    for(int i=0;i<(int)ans.size();i++)
		fout2<<conv.to_bytes(ans[i].second)<<endl;
}

bool splitorForCh::isCh(int x)
{
	if(x>=0x4e00&&x<=0x9fa5)
		return true;
	return false;
}

void splitorForCh::fun(wstring& wd)
{
	for(int i=0;i<wd.size();i++)
	{
		for(int j=2;j<=limit;j++)
		{
			if(i+j>wd.size()) break;
			ma[wd.substr(i,j)]++;
		}
	}
}

void splitorForCh::split(wstring& ws)
{
	ws+=L" ";
	int l =0;
	for(int i=0;i<ws.size();i++)
	{
		if(!isCh(ws[i]))
		{
			if(i-l>1)
			{
				wstring wd = ws.substr(l,i-l);
				fun(wd);
			}
			l = i+1;
		}
	}
}

void splitorForCh::read(string filepath)
{
	wstring_convert<codecvt_utf8<wchar_t>> conv;
	ifstream fin(filepath);
	while (!fin.eof())
	{
		string line;
		getline(fin, line);
		wstring ws = conv.from_bytes(line);
		split(ws);
	}
}

splitorForCh spch;

int main(int argc, char *argv[])
{
	clock_t start=clock();		
	spch.splitTxtCh(300,2,"Chs_in.txt","Chs_subword.txt");
    clock_t end=clock();		
	double endtime=(double)(end-start)/CLOCKS_PER_SEC;
	cout<<"Total time:"<<endtime<<endl;		
    printf("CH\n");
	return 0;
}
