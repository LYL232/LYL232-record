#include <cstdio>
#include <string>
#include <ctime>
#include <iostream>
#include <unistd.h>/* gethostname */
#include <netdb.h> /* struct hostent */
#include <arpa/inet.h> /* inet_ntop */

using namespace std;

bool GetHostInfo(std::string &hostName, std::string &Ip) {
    char name[256];
    gethostname(name, sizeof(name));
    hostName = name;

    struct hostent *host = gethostbyname(name);
    char ipStr[32];
    const char *ret = inet_ntop(host->h_addrtype, host->h_addr_list[0], ipStr, sizeof(ipStr));
    if (nullptr == ret) {
        std::cout << "hostname transform to ip failed";
        return false;
    }
    Ip = ipStr;
    return true;
}

int main(int argc, char *argv[]) {
    std::string hostName;
    std::string Ip;

    bool ret = GetHostInfo(hostName, Ip);
    if (ret) {
        std::cout << "ip: " << Ip << ", hostname: " << hostName << std::endl;
    }
    return 0;
}

