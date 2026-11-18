/* ══════════════════════════════════════════════════════════════════
   Figure 10 — The Neuroconnectionist Trifecta & opportunistic model
   experiments
   Perceptual representations reflect three interacting pressures:
   1. Sensory Diet (Input: e.g., dogs-only, cats-only, dogs+cats, or
      paired vision + language in CLIP)
   2. Computational Architecture (2D convolutional kernel vs. global
      self-attention head)
   3. Computational Objective (Supervised, Autoencoder, Contrastive, CLIP)
   ══════════════════════════════════════════════════════════════════ */
DL4HL.register('trifecta_explorer', function (root, cfg) {
  var M = DL4HL, C = M.C;
  M.frame(root, cfg, {
    fig: 10,
    title: 'Types of deep learning models: sensory diet, architecture, & objective',
    hint: 'Choose which Trifecta factor to vary &middot; inspect the Input, Architecture, &amp; Model-Zoo schematic &middot; press play or drag the 3D hypersphere'
  });

  var THUMB_URIS = {
    dogs: [
      'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAcFBQYFBAcGBgYIBwcICxILCwoKCxYPEA0SGhYbGhkWGRgcICgiHB4mHhgZIzAkJiorLS4tGyIyNTEsNSgsLSz/2wBDAQcICAsJCxULCxUsHRkdLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCz/wAARCAAwADADASIAAhEBAxEB/8QAGgAAAgMBAQAAAAAAAAAAAAAABAUDBgcIAP/EADEQAAIBAwIDBgUDBQAAAAAAAAECAwQFEQAhBhIxBxMiQVFhFHGBobEjMpEVFkLR4f/EABgBAQEBAQEAAAAAAAAAAAAAAAQDAgUG/8QAIREAAgICAgIDAQAAAAAAAAAAAQIAEQMSBCFBoTGBsfD/2gAMAwEAAhEDEQA/AKZy1MsyLCfEp5woGc+f40Z+qkOJCGmUArGrDLEdQfTz0s72pQgQVBiEnhkRW3Yg+fsD030yp1V5UjkcM4HKzsBsMb499eayCqmDCaKGGpqkTvJJsMC2Bnrk4B8zosUsVMB8THJA5AHKykHPn00z4SpJqioppaNMU6vKskr9VCD8knVsktMT2+oSTmklNScSPv4OTOB9To7EK+jGuo/Fxdl2MzoRQU5buGRFOWyBuSSN86gqI4VJ5oi0vIVVum56npvpXfeIZaG/V8D0wjETKCAOUeo26A6Lp6lrqkaUrPJ07s4Jc825A9cHb11dsDpTHzCupU1ArUkPeCNUXPLkMq5Ow6gf7204t1CjVJJbDqrY9yFJHXruPxrySR2uCGJX56iRBmUOI+8yei+xzjPn5aDeRRDIedhLzB1yTzKfLfWnYk2JPpSJcuDLZNfeEf6dS3GpoqoSswkhblL+INg+xyNvbVs48tt4snCtOLeEa4SAp4jhQx8x9BpH2avVQ193rLfDzmloGqI6dhjMmPCv5/gagsPH1746tN5huEAlp6JEaGcRcjd5zbrtsTjfVW4YbfK/fdj0T+TpjkFSqj+8TMeMGrFtbT3V4XuFV3aSGIYBKliPscaS2Hi+58LUqT0Miczu4AcHYYGSD5ak41rmqeIKimzlIJCB88D8YA1Wq4/qxwhspCgUfPqfuTrpYcQOMK4+e4bMwZjU0FpHVi8SNyI+ctkeIe/8baaRQd5Sx1R8ck+CqjJYMP8Avl89OeFuzK88URmsUQUdtk/bUVBJaQA/4oOo98ga0CP+3uy+2MaZvjbtIrL8TKBzNtkhQNlG3T76AcViz0IMDyYNwBwVdKS8RXeqr5LXTzQdy1K+A8yYwRg/t6AgnfbppxU2i322re3UFero7c00QxzhSdyAPXffWUXXtCudTVA8pjkOAzOSeXJPT0OhOHrtVW3jCnvLFqhwcyjOQ653B+2NUXIFXUym19yiX2irk4kuNXWW+WBPiJGCOMYyxKg/TGqxM/NM7N1JzrrrtE4bpOJuB5Lza4FkSWATAKPENs/b01yZU0EwrpI5EaIq2GDDBB+WnYMjMDuKPr6lX1oamf/Z',
      'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAcFBQYFBAcGBgYIBwcICxILCwoKCxYPEA0SGhYbGhkWGRgcICgiHB4mHhgZIzAkJiorLS4tGyIyNTEsNSgsLSz/2wBDAQcICAsJCxULCxUsHRkdLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCz/wAARCAAwADADASIAAhEBAxEB/8QAGwAAAgMAAwAAAAAAAAAAAAAABQcDBAYAAQL/xAAvEAACAQIEBQIGAQUAAAAAAAABAgMEEQAFEiEGEzFBUQdhFCJxgZGhFjIzkrHw/8QAGQEBAQADAQAAAAAAAAAAAAAAAwQAAgUB/8QAHxEAAgIDAQADAQAAAAAAAAAAAQIAAxESMSEEE0FR/9oADAMBAAIRAxEAPwBMw0tQNaQyxCGc/M7Dpb/v3iM08tMGvGZEcWRugvte30v+8HcoyvMs20qiimQgNrYWQqT298CaxaqKpmpKhQssL6UjJstu5GBDZMBe9gCeI85iQyi9gT5GCuRZTWZxVw0tO0RaVtCmWQIpYmwGo9z4x6p4pah41dVIeQWt8wG/cYYnC3DyZdFVZhU0TRShHSlBYxDm2Nje+wvt9d+2PXs1Eesfa+hEz+e8AZpwrlkhzeRUkCmSOOBjIjHUFtqG1xcE+1sZ6jyNIW+JrIxPCyagig7E+cbn+QVEfp4EzfLFimqKvkxSysxciOKxO53tdVv3tgJS5vFByoaQgTq2osbb97b9Re3XxjMtwwr8BtU8EmqM1qWiblsym2qw264D18hrIfi5SedcIoNz9vphjT5fK6GJaCInppCgbYFycHTSl6mukeGBDzLoFQ2Ft7dBb87e+JanA7DrALedgXh3geepenqJ546SmdQTM8hUx3YCwsRckHp9MPWj4cNdBDQPd40AR4zZjZdgbn2thFiPMM6zY1E3MFKHIi1AhD777E+Th/8ACVYwmotU6vIihZLN2A6j7f6xtYpOuxnWoGinz2KX12pJKTiXLcpUiCmiy8SxIoADuZG1E/4j8DGF4UyamzQ1BqpmhZQBEYyCCx2t5w7PUXgubjHi96+SfQsES08dz2BJP7Y/rA7L/Sn4ZtRzBFRTcctbEEdN8bPZ1QZy2bLS6pLf3NRO1ugF8VAlPnNc1AWWZkYa4w3fqQfFhvi3NlBm182KMGQAMqMykjz8rC2OZZk5yjOHr6eCPmGJoViIsqAkb7bnp5xHqP7H+MtVNm59xDddDluYZIuUlwppauMQgKToIW7KPYr1+2C9LFRU9d8JTaFkWLWUTYLYG1/zgGzTIjvTI8FaTr56lSASbkBSCADjrI6SbLs4FW85mUseYtralP8AVbx7YcOqqFEpe7azI5C9dXoa+pHNUHmbLbp3xGJNcgMkxCk3HWzYhr6VKiqjlTmCWw5hsNLhRYG3Y2Avvv7YgeCSchQSh7A+cE2MyfC/s//Z',
      'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAcFBQYFBAcGBgYIBwcICxILCwoKCxYPEA0SGhYbGhkWGRgcICgiHB4mHhgZIzAkJiorLS4tGyIyNTEsNSgsLSz/2wBDAQcICAsJCxULCxUsHRkdLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCz/wAARCAAwADADASIAAhEBAxEB/8QAGgAAAgMBAQAAAAAAAAAAAAAABQYCBAcDAf/EADMQAAIBAgQFAQYEBwAAAAAAAAECAwQRAAUSIQYTMUFRcQcUImGBoRUyQtEXUnKCkbHx/8QAGQEBAAMBAQAAAAAAAAAAAAAAAgEEBQMG/8QAHxEAAgICAwADAAAAAAAAAAAAAQIAEQMhBBITMUFR/9oADAMBAAIRAxEAPwB0QrG2lVUeAH6/8xLS+hiqEHbcv/vCN/EPhyON5Y1r6pNWkSRwW7eCfGGDI+I8q4mozLl88wZVDssq6WG9gTba3Xp4x5lsTqLYTWDqdCEKueGmieWonhgTYapJgAcQoK3L6+Z0pMwp5mSzEQTh9PqAdsZn7TMseHiOjrhVtLDPIYlQncMFG47W3++BHCLVFP7RcrFNYNPrhkC7Bksbg/MHf6Y7rxlbH3BgOQ3VTcllCLcaRp3udyfH2Bx5HViQlQwQ3sN7YRuN+N5eE6yCjoqOOSaRAxnkuY4wTbp+o7H5DCxTe0HNaOsiNYRW00p+OPkrG4uPzIw6+bHtgJx8jL2ERyKDRi/HQqkFbJS1Rkp6ecxxtsrPfuB2Ft8Csh40zPhbMHWlWKaGSQrLCw2kAbpfqO9j88X6+l90kbKzCzO83Ji0kq17GwBv31emAOb8N5jlNaU93ZmhI1hPjCNuwF+5spJ8Y10Aaw+7lR9UVmj8YcS0md1eXLRqFheL3izmxDN5PUadIHrhazTPKaPiqnr8kQ0Zo4UcIfyvIL6hcHcEnc98UqfJ6qSmkhq3COkbs5BGpNJ1WI6gdcUqnL6lcshEKiZmk5KCIXZv3wMeNF0ImyuVqpoHEPE1FxVwbTusOitknSORB+lVOplv/KTY/UeMB4638G4mpJs4pEipjzCyx/EFXlso2G/f7YBNlE1Ll8LlSEjVBUOjagC1ypFvHQ/XFaqbMK+oeaV5J1ik5Ksxvc3sAP8AF8BcK/CnW4vQhdjeppud5DNW51QVNPVFZopY5VuoFreftg02Ux8jMpKlkmeWfWq2KgAAqL23NwbH5DHeq5SzVEqBTFHGEVIxZiSDt8unX0xb5iowdpC0gjS8ajZ7m3Xzv9bYzPU0BL3QXAVDwzlq1FTWNRrLKzmMvrttckgD+7v++IZdw1kuWUULzISkdQZ4oSbJdoyoFyb2+En1weNO8RjiRNPxkshuzgkd73v64pyoKmnMCFpzSyKpsdQstrgHubEi3fEjMx1cgqPyKYos0zTNa/kSJT0R5cSALdNQsBcWPTci3nHaTh+QVa5Q8iy0cSmokmVLEyrY6v6bN37DDIKR4Pd4lgPLEl9JGkAk283vY74vingpqoo5jKEBJXU9eg8dML1P1J6if//Z'
    ],
    cats: [
      'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAcFBQYFBAcGBgYIBwcICxILCwoKCxYPEA0SGhYbGhkWGRgcICgiHB4mHhgZIzAkJiorLS4tGyIyNTEsNSgsLSz/2wBDAQcICAsJCxULCxUsHRkdLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCz/wAARCAAwADADASIAAhEBAxEB/8QAGgAAAwEBAQEAAAAAAAAAAAAABAYHBQMCCP/EAC0QAAEDAwMDAgYCAwAAAAAAAAECAwQABREGEiETMUEUUQciMmFxkRUjQoGx/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AKDq1gO2F8Ac7ajGoZAj2iW20tSp6khLTbffkjx98gfurrfEBVqdB7bTUYtN0n2tcmbd2Iv8erqKdByXmm9wCVgdlAnAwOeQaDparRcGra0mRb0JUCkOJCwoJxg5BHj7dxg1TpS7ZfbI7IhPNvJxtWkcKQr2UO4NTK46qlrgBNjSmU+8ha2Ax/YOmFBPU47cnsecgiiXOto+THZZdS5dZMfDiXgpe9Lac7Tg8DlSs5J3LA7A0DHZrBuYWggnB4zXqRp7Cj8lcbX8R7fbIHqbla5re88hlKXEj8nIp4ssyBqiyM3a3hZjP529RO1QIOCP2KAm5o3wHB7pNJCbBE1XpqRbVrVFkAhKJCRnapJBSSPIyBkU8zXUenLZUN6xwPJqcs3Wda75KgRWjyrLjhGS0T2wOx96A74e6Uc0jZJJuiWjLU+4pbiTu3IB+XBx5wVflVLcd6JdtbXK9NurdYUSy0pxe7elP1lCf8UhXy58kU2XS1zNbfD9+EZMizPOJI3duqU54PnpqOCcckVNNHRZejZ3p53QkpuEjpLjMciOdp2KBxg7iNpA+xNBoaVszutdQTbPHSYsCM+UyX8ZAGcgNn3Psa+golvi2yCxBhspZjR0BtttPZKR2pK0PBRaQzEQkIOCtQAxlROT/wBx/qn1z6qCVT9Vwol0kOTH9iGsJGO/Azx7nmh7ZKRcXnri80A4+N3AxwBwKmN6lSrrqBTTawIbq0uKJ8kAcfuni1TUtoaZ3cIwSfxQb9u9c/afS73YpPAUPqHnvWhp/RMCLdUXCUESHWnOqwnYEhpeOSMfk/uh412QMciteLdEZBKsCgJQj0mr0t4wl1JWj7jzTY54pOuUluQuJLQ6EuxnNw+47Efo01JkpW2k57ig/9k=',
      'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAcFBQYFBAcGBgYIBwcICxILCwoKCxYPEA0SGhYbGhkWGRgcICgiHB4mHhgZIzAkJiorLS4tGyIyNTEsNSgsLSz/2wBDAQcICAsJCxULCxUsHRkdLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCz/wAARCAAwADADASIAAhEBAxEB/8QAGgAAAgMBAQAAAAAAAAAAAAAABQYCBwgEA//EACwQAAIBAgQFBAICAwAAAAAAAAECAwQRAAUSIQYTIjFBFFFxkWGBBxVCoeH/xAAWAQEBAQAAAAAAAAAAAAAAAAAFBAP/xAAgEQACAgIBBQEAAAAAAAAAAAABAwACBBESEyEiMUFC/9oADAMBAAIRAxEAPwDPdNDVV8609HTyTzMNkiUsx+AMeseX11LUOJ6SohMY6w8bLpv2vfEMozWpyTNqfMaNtM9O+pb9j7g/gi4xpfhDizKuK+FzU8qPXYJPG4DFWH+J9x5GJ3N6fsdpupXU9e5nOgp5HqhHHGzyOQFAFyfjDanAPEQCVP8AXSWdx0m2r6xeMmVZLHVQ1MGX0yVEC6Y2jjVdP1jryWVHhqpKoWmCsYgTsoHi3v5vgpmabW40ETVi8a87Sv8AIOFKqdKf1DemEeoHV5NyQMHpuB8tqapFr5JZAlm0qQouPOHqDKxHSXaO8dQLOTv1EDf7ws1TtThWka5iJ6j5W9j+wf8AWImA11b7LVsN91B7TJuLA/iF53z6upo53SOSm1Mi76yGFtv3hAKsPBxaX8P5I8S12eT3RNPIiBB6jcFjt+hh7NsAi24JhVNn11LfSLTTos4VVkFg0iFwGv5C7jEquo9BJRSuQqyg3kj2jB2uDfcEG+x7gjEIYY6ioQ9Q5R6mYiMDY7Asbn9Y682qUiq6eikgapkGl3p0GoqD2ue1/wAYAqBxjlj56jXTzSPlJXWJEY3Vgdj7YXc/olaCoI6SkpW1vcC4+jbB3JqDMY6GgpqpIlhkZpTGN3QX1BSfn67YF8QViLV5jTLp5iTc1ixFx0A+Pzi1y9IBMiSeLyKntKuyzhn1rqlNSqx99AAGGmXhwZWtOhLcxN2IJtYHtpGw3Pzjv4ZkC0sfLm30mwFtr+/ucF6uJ51ISp5LXuSUDX+8bZdGOGvkxxGUSdiBZJI4qqJ3WCRG2Ebxhy5tfuRtgxQ5fPPWGreQapIho17bXtf8C+Bs2WVj1kTq9O4jlDKDqHztgvNDmstWjN6UKqaQodrW8X2wdRDP0Ig11NeJEYRK1PyeU3MkQHSCfNrb4rapyuqn4kzBJZAJaia8hXcCNQAd/wAnYfBw+ZbSzmojkqnh1pfSFvb/ALgQconhkrXjzGA1E8rOzNA2wPZRv2GLG0ZdYFRI0Xoq5NjP/9k=',
      'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA8Q1JFQVRPUjogZ2QtanBlZyB2MS4wICh1c2luZyBJSkcgSlBFRyB2NjIpLCBxdWFsaXR5ID0gOTAKAP/bAEMABwUFBgUEBwYGBggHBwgLEgsLCgoLFg8QDRIaFhsaGRYZGBwgKCIcHiYeGBkjMCQmKistLi0bIjI1MSw1KCwtLP/bAEMBBwgICwkLFQsLFSwdGR0sLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLP/AABEIADAAMAMBIgACEQEDEQH/xAAaAAADAQEBAQAAAAAAAAAAAAAFBgcDBAEC/8QAMxAAAgECBAQDBgUFAAAAAAAAAQIDBBEABRIhBhMxQVFhcQcUIkKBkRUXYsHwIyQzsdH/xAAYAQADAQEAAAAAAAAAAAAAAAABAgMABP/EABsRAAMBAAMBAAAAAAAAAAAAAAABAhEhMUED/9oADAMBAAIRAxEAPwAHmJQtTli6oFY/Dsb3269v2xrQ5YsjywtLFT0yskkiSsBIpDXAHiNtiP2wLzOpq3FGHpuQrSaNRYHY7WsPXGWeZi9SKGspeWje78rlg6mAvszHxuL+WIVCqlpRZg7+7ZXS5slRJM0kqggMdgEuW38d2vjI57JX8R0+uFYqKphkWLSLNYC4Y+p7Yn1dmlc8Bi1f5A0TnqbEWODlDmgkrsngSVqmKmBUhowrJ02Ldwd8C/jL6RSbaWA/ikFXgnUXkiH3Hf8AnlhkzijoG4NijSnVfe4Y21AmxARnO3npH3wHzyHm7RnW6fFYbnbHyEnWWjy+Wp/t3pA1MZFuLuQCl/03cb+QxK1rQs+jA8Mc9XRrKARCxka/Q2BsPv8A6wBzaKlSzNA8UbsWMlho02PSx23wRr6s00MTAbGddZJ6qRY/TB7JeHqjibPqSmiZaRoX1Sum4EXcDsR/3HUKmIdZktZQUlDXTQuYswiZ4SFLFWDFdJHjcWwVy9xC0NS0UMayfIragCOx7Xxba/h7Kc8hHC4oaqBcspysdYDpClrgLf5tQuxt0+uIzxLE/Dr1FBNyylMNHM0E7g7WI79vTBN1wxMmzqKKsqaSdCyxSuI3VrMNzbAufinMXkhKzC0AKx/pF74F1ZtK0nM1FySW9cc6oxjLjoMFpMTSr5+inL4ZNN9TaCPHuMVf2bZP+K8CQ1yStTVsc8kaTIeqgjY+V74lOagtlsRuNCkffpi1exdo/wAuxHt/SqpQfqQcJ6BvkR/aBV5vwaq1sU8olqGs8vOaxt2sdvTbElzfO5c7dJmLQkA69DaSvmT81/PFD9u2dfiedw0cEg5NHcWXf4z3+2JPLakpCzD43NzfqTjSOzgzALJUqikeJJ7YyYIkDxcyzKbb98eMTJVhrdBc2xjUkiofrcbk+f8ADhxT/9k='
    ]
  };

  var thumbImgs = { dogs: [], cats: [] };
  ['dogs', 'cats'].forEach(function (kind) {
    THUMB_URIS[kind].forEach(function (uri) {
      var im = new Image();
      im.onload = function () { if (ready) drawAll(); };
      im.src = uri;
      thumbImgs[kind].push(im);
    });
  });

  var VALID_OBJ = ['supervised', 'autoencode', 'contrastive', 'clip'];
  var varyAxis = cfg.vary || 'task';
  var taskMode = VALID_OBJ.indexOf(cfg.objective) >= 0 ? cfg.objective : 'contrastive';
  var archMode = 'cnn';   // 'cnn' | 'vit'
  var dietMode = 'both';  // 'dogs' | 'cats' | 'both'
  if (varyAxis === 'arch') { taskMode = 'supervised'; dietMode = 'both'; }
  if (varyAxis === 'input') { taskMode = 'contrastive'; archMode = 'cnn'; }

  var prog = 1.0; // 0..1 training progression
  var yaw = -0.32, pitch = 0.24, dragOrigin = null;
  var ready = false;

  /* 24 instances: 12 Dogs (family 0, blue) + 12 Cats (family 1, orange) */
  var N = 24, initPts = [], initTwins = [], fam = [];
  (function buildInit() {
    var rng = M.rng(cfg.seed || 53);
    for (var i = 0; i < N; i++) {
      fam.push(i % 2);
      var v = M.normVec([rng.normal(0, 1), rng.normal(0, 1), rng.normal(0, 1)]);
      var tw = M.normVec([v[0] + rng.normal(0, 0.50), v[1] + rng.normal(0, 0.50), v[2] + rng.normal(0, 0.50)]);
      initPts.push(v);
      initTwins.push(tw);
    }
  })();

  var SUP_CENTERS = [
    M.normVec([0.76, 0.34, 0.55]),   // Dog prototype
    M.normVec([-0.76, -0.28, 0.58])  // Cat prototype
  ];

  function targetPositions(mode) {
    var rng = M.rng(909 + (archMode === 'vit' ? 17 : 0));
    var pts = [], twins = [];
    var spreadSup = archMode === 'vit' ? 0.13 : 0.19;
    for (var i = 0; i < N; i++) {
      var f = fam[i], u = i / N;
      if (mode === 'supervised') {
        var c = SUP_CENTERS[f];
        var p1 = M.normVec([c[0] + rng.normal(0, spreadSup), c[1] + rng.normal(0, spreadSup), c[2] + rng.normal(0, spreadSup)]);
        pts.push(p1); twins.push(p1);
      } else if (mode === 'autoencode') {
        var lon = (u - 0.5) * Math.PI * 1.75;
        var lat = (f === 0 ? 0.34 : -0.34) + 0.14 * Math.sin(u * 8);
        var p2 = M.normVec([Math.cos(lat) * Math.sin(lon), Math.sin(lat), Math.cos(lat) * Math.cos(lon)]);
        pts.push(p2); twins.push(p2);
      } else if (mode === 'contrastive') {
        var phi = Math.acos(1 - 2 * (i + 0.5) / N);
        var theta = Math.PI * (1 + Math.sqrt(5)) * i;
        var p4 = M.normVec([Math.sin(phi) * Math.cos(theta), Math.cos(phi), Math.sin(phi) * Math.sin(theta)]);
        var tw4 = M.normVec([p4[0] + rng.normal(0, 0.05), p4[1] + rng.normal(0, 0.05), p4[2] + rng.normal(0, 0.05)]);
        pts.push(p4); twins.push(tw4);
      } else {
        /* CLIP: paired image + text vectors converge around semantic caption regions */
        var c5 = SUP_CENTERS[f];
        var p5 = M.normVec([c5[0] + rng.normal(0, 0.22), c5[1] + rng.normal(0, 0.22), c5[2] + rng.normal(0, 0.22)]);
        var cap = M.normVec([p5[0] + rng.normal(0, 0.06), p5[1] + rng.normal(0, 0.06), p5[2] + rng.normal(0, 0.06)]);
        pts.push(p5); twins.push(cap);
      }
    }
    return { pts: pts, twins: twins };
  }

  /* ─── DOM (Trifecta Schematic first: left on desktop, above on narrow screens) ─── */
  var split = M.split(root, 1.18);
  M.mini(split.a, 'trifecta schematic: sensory diet, architecture, & model zoo');
  M.mini(split.b, 'latent hypersphere: how representations organize');
  var hf = function (w) { return M.clamp(w * 0.78, 324, 368); };
  var schemStage = M.stage(split.a, { height: hf, ref: 450, label: 'Visual depictions of dog and cat sensory diets, convolutional vs attention architectures, and controlled model-zoo experiments' });
  var sphereStage = M.stage(split.b, { height: function () { return schemStage.h || 334; }, ref: 390, label: '3D latent hypersphere showing how supervised, autoencoder, contrastive, and CLIP objectives arrange dog and cat representations' });

  M.legend(root, [
    { color: C.blue, label: 'dog exemplars' },
    { color: C.orange, label: 'cat exemplars' },
    { color: C.green, label: 'supervised label / paired text anchor' }
  ]);

  var tr = M.transport(root, {
    n: 61, fps: 30, aria: 'Training progression from random initialization to learned geometry',
    fmt: function (k) { return 'training ' + Math.round((k / 60) * 100) + '%'; },
    onFrame: function (k) { prog = M.ease(k / 60); drawAll(); }
  });
  tr.show(60);

  var row1 = M.row(root);
  var varyT = M.toggle(row1, {
    label: 'Vary 1 factor', value: varyAxis,
    options: [
      ['task', 'Task Objective'],
      ['arch', 'Architecture'],
      ['input', 'Sensory Diet']
    ],
    onChange: function (v) {
      varyAxis = v;
      if (v === 'arch') { taskMode = 'supervised'; dietMode = 'both'; }
      else if (v === 'input') { taskMode = 'contrastive'; archMode = 'cnn'; }
      else { dietMode = 'both'; }
      rebuildSubControl();
      tr.show(60);
      drawAll();
    }
  });
  M.spacer(row1);
  M.resetButton(row1, function () {
    tr.pause();
    varyAxis = 'task'; taskMode = 'contrastive'; archMode = 'cnn'; dietMode = 'both';
    yaw = -0.32; pitch = 0.24;
    varyT.set('task', true);
    rebuildSubControl();
    tr.show(60); drawAll();
  });

  var row2 = M.row(root);
  var read = M.readout(root);

  /* Coupled sub-control row: shows ONLY the active factor's levels so held factors never conflict */
  function rebuildSubControl() {
    row2.innerHTML = '';
    if (varyAxis === 'task') {
      M.toggle(row2, {
        label: 'Objective (varied)', value: taskMode,
        options: [
          ['supervised', 'Supervised'],
          ['autoencode', 'Autoencoder'],
          ['contrastive', 'Contrastive'],
          ['clip', 'CLIP (Vision + Text)']
        ],
        onChange: function (v) { taskMode = v; tr.show(60); drawAll(); }
      });
      M.label(row2, '&middot; Held: Dogs+Cats &amp; CNN/ViT');
    } else if (varyAxis === 'arch') {
      M.toggle(row2, {
        label: 'Architecture (varied)', value: archMode,
        options: [
          ['cnn', 'CNN (2D Conv)'],
          ['vit', 'Transformer (Self-Attn)']
        ],
        onChange: function (v) { archMode = v; tr.show(60); drawAll(); }
      });
      M.label(row2, '&middot; Held: Dogs+Cats &amp; Supervised');
    } else {
      M.toggle(row2, {
        label: 'Sensory Diet (varied)', value: dietMode,
        options: [
          ['dogs', 'Dogs only'],
          ['cats', 'Cats only'],
          ['both', 'Dogs + Cats']
        ],
        onChange: function (v) { dietMode = v; tr.show(60); drawAll(); }
      });
      M.label(row2, '&middot; Held: CNN &amp; Contrastive');
    }
  }
  rebuildSubControl();

  M.pointer(sphereStage.cv, {
    down: function (p) { dragOrigin = { x: p.x, y: p.y, yaw: yaw, pitch: pitch }; return true; },
    drag: function (p) {
      if (!dragOrigin) return;
      yaw = dragOrigin.yaw + (p.x - dragOrigin.x) * 0.012;
      pitch = M.clamp(dragOrigin.pitch + (p.y - dragOrigin.y) * 0.012, -1.1, 1.1);
      drawAll();
    },
    up: function () { dragOrigin = null; }
  });

  function drawSpherePanel() {
    var ctx = sphereStage.ctx, s = sphereStage;
    if (!s.w) return;
    M.clear(ctx, s);
    var R = Math.min(s.w, s.h - 24 * s.sc) * 0.40;
    var cx = s.w * 0.5, cy = s.h * 0.54;
    M.drawSphere(ctx, cx, cy, R, yaw, pitch, s);

    var targ = targetPositions(taskMode);
    var famCols = [C.blue, C.orange];

    /* Supervised target attractors on the sphere (drawn behind/around points with explicit labels) */
    if (taskMode === 'supervised') {
      [['Label: "Dog"', SUP_CENTERS[0], C.blue], ['Label: "Cat"', SUP_CENTERS[1], C.orange]].forEach(function (item) {
        var sp = M.sphereProject(item[1], cx, cy, R, yaw, pitch);
        ctx.strokeStyle = M.rgba(item[2], 0.55);
        ctx.setLineDash([3 * s.sc, 2 * s.sc]);
        ctx.lineWidth = 1.3 * s.sc;
        ctx.beginPath(); ctx.arc(sp.x, sp.y, 17 * s.sc, 0, Math.PI * 2); ctx.stroke();
        ctx.setLineDash([]);
        M.star(ctx, sp.x, sp.y, 6.2 * s.sc, C.green, C.surface);
        M.pill(ctx, s, item[0], sp.x, sp.y - 16 * s.sc, { align: 'center', color: item[2], size: 8.0 });
      });
    } else if (taskMode === 'clip') {
      [['"a photo of a dog"', SUP_CENTERS[0], C.blue], ['"a photo of a cat"', SUP_CENTERS[1], C.orange]].forEach(function (item) {
        var sp2 = M.sphereProject(item[1], cx, cy, R, yaw, pitch);
        M.pill(ctx, s, item[0], sp2.x, sp2.y - 18 * s.sc, { align: 'center', color: item[2], size: 7.8 });
      });
    }

    for (var i = 0; i < N; i++) {
      var f = fam[i];
      if (dietMode === 'dogs' && f !== 0) continue;
      if (dietMode === 'cats' && f !== 1) continue;

      var curP = M.normVec([
        M.lerp(initPts[i][0], targ.pts[i][0], prog),
        M.lerp(initPts[i][1], targ.pts[i][1], prog),
        M.lerp(initPts[i][2], targ.pts[i][2], prog)
      ]);
      var curT = M.normVec([
        M.lerp(initTwins[i][0], targ.twins[i][0], prog),
        M.lerp(initTwins[i][1], targ.twins[i][1], prog),
        M.lerp(initTwins[i][2], targ.twins[i][2], prog)
      ]);

      var pp = M.sphereProject(curP, cx, cy, R, yaw, pitch);
      var col = famCols[f];
      var front = pp.z >= 0;

      if (taskMode === 'contrastive' || taskMode === 'clip') {
        var pt = M.sphereProject(curT, cx, cy, R, yaw, pitch);
        ctx.strokeStyle = M.rgba(col, front ? 0.55 : 0.2);
        ctx.lineWidth = 1.2 * s.sc;
        ctx.beginPath(); ctx.moveTo(pp.x, pp.y); ctx.lineTo(pt.x, pt.y); ctx.stroke();
        if (taskMode === 'clip') {
          M.dot(ctx, pt.x, pt.y, 3.3 * s.sc, C.surface, M.rgba(C.green, front ? 0.92 : 0.35), 1.5 * s.sc);
        } else {
          M.dot(ctx, pt.x, pt.y, 2.7 * s.sc, M.rgba(col, front ? 0.75 : 0.25));
        }
      }
      M.dot(ctx, pp.x, pp.y, (front ? 3.8 : 2.7) * s.sc, M.rgba(col, front ? 0.92 : 0.30), front ? C.surface : null, 0.9);
    }

    var caption = {
      supervised: 'Supervised: points move toward explicit class labels ("Dog", "Cat")',
      autoencode: 'Autoencoder: points organize along smooth pose & appearance manifolds',
      contrastive: 'Contrastive: pull augmented views together, push instances apart',
      clip: 'CLIP: align image (\u25cf) and text caption (\u25cb) embeddings on the sphere'
    }[taskMode];
    M.pill(ctx, s, caption, 8 * s.sc, 13 * s.sc, { size: 7.9 });
  }

  function drawThumb(ctx, kind, idx, x, y, sz, borderCol, s) {
    var img = thumbImgs[kind][idx % thumbImgs[kind].length];
    ctx.save();
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, sz, sz, 4 * s.sc); else ctx.rect(x, y, sz, sz);
    ctx.clip();
    if (img && img.complete && img.naturalWidth > 0) {
      ctx.drawImage(img, x, y, sz, sz);
    } else {
      ctx.fillStyle = M.rgba(borderCol, 0.18);
      ctx.fillRect(x, y, sz, sz);
    }
    ctx.restore();
    ctx.strokeStyle = borderCol;
    ctx.lineWidth = 1.25 * s.sc;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, sz, sz, 4 * s.sc); else ctx.rect(x, y, sz, sz);
    ctx.stroke();
  }

  function drawPhotoGroup(ctx, kind, gx, gy, gw, gh, col, active, s) {
    ctx.save();
    ctx.fillStyle = active ? M.rgba(col, 0.07) : M.rgba(C.border, 0.20);
    ctx.strokeStyle = active ? M.rgba(col, 0.72) : C.border;
    ctx.lineWidth = (active ? 1.2 : 0.9) * s.sc;
    if (!active) ctx.setLineDash([3 * s.sc, 2 * s.sc]);
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(gx, gy, gw, gh, 4 * s.sc); else ctx.rect(gx, gy, gw, gh);
    ctx.fill(); ctx.stroke();
    ctx.setLineDash([]);

    ctx.globalAlpha = active ? 1.0 : 0.30;
    var gap = 5 * s.sc;
    var sz = Math.min(gh - 8 * s.sc, (gw - 12 * s.sc - 2 * gap) / 3);
    var totalW = 3 * sz + 2 * gap;
    var x0 = gx + (gw - totalW) * 0.5;
    var y0 = gy + (gh - sz) * 0.5;
    for (var i = 0; i < 3; i++) {
      drawThumb(ctx, kind, i, x0 + i * (sz + gap), y0, sz, col, s);
    }
    ctx.restore();
  }

  function drawConvSchematic(ctx, x, y, w, h, active, s) {
    ctx.fillStyle = active ? M.rgba(C.blue, 0.07) : C.surface;
    ctx.strokeStyle = active ? C.blue : C.border;
    ctx.lineWidth = (active ? 1.6 : 1.0) * s.sc;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, 4 * s.sc); else ctx.rect(x, y, w, h);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = active ? C.blue : C.muted;
    ctx.font = M.font(s, 8.0, 600, 'sans');
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('CNN: 2D Convolution', x + 6 * s.sc, y + 5 * s.sc);

    /* 5x5 input grid + highlighted 3x3 sliding kernel -> 3x3 feature map */
    var cell = Math.min((h - 24 * s.sc) / 5.2, (w * 0.42) / 5.2);
    var gx = x + 8 * s.sc, gy = y + 19 * s.sc;
    for (var r = 0; r < 5; r++) {
      for (var c = 0; c < 5; c++) {
        var inKernel = r >= 1 && r <= 3 && c >= 1 && c <= 3;
        ctx.fillStyle = inKernel ? M.rgba(C.blue, 0.25) : C.codeBg;
        ctx.strokeStyle = inKernel ? C.blue : C.border;
        ctx.lineWidth = (inKernel ? 1.1 : 0.6) * s.sc;
        ctx.fillRect(gx + c * cell, gy + r * cell, cell, cell);
        ctx.strokeRect(gx + c * cell, gy + r * cell, cell, cell);
      }
    }

    var fx = gx + 5 * cell + 20 * s.sc, fy = gy + cell;
    for (var r2 = 0; r2 < 3; r2++) {
      for (var c2 = 0; c2 < 3; c2++) {
        var isCenter = r2 === 1 && c2 === 1;
        ctx.fillStyle = isCenter ? C.blue : M.rgba(C.blue, 0.15);
        ctx.strokeStyle = C.border;
        ctx.lineWidth = 0.7 * s.sc;
        ctx.fillRect(fx + c2 * cell, fy + r2 * cell, cell, cell);
        ctx.strokeRect(fx + c2 * cell, fy + r2 * cell, cell, cell);
      }
    }

    /* Projection rays from 3x3 kernel corners to active output cell */
    ctx.strokeStyle = M.rgba(C.blue, 0.65);
    ctx.lineWidth = 1.0 * s.sc;
    ctx.beginPath();
    ctx.moveTo(gx + 4 * cell, gy + 1 * cell); ctx.lineTo(fx + 1.5 * cell, fy + 1.5 * cell);
    ctx.moveTo(gx + 4 * cell, gy + 4 * cell); ctx.lineTo(fx + 1.5 * cell, fy + 1.5 * cell);
    ctx.stroke();
  }

  function drawAttnSchematic(ctx, x, y, w, h, active, s) {
    ctx.fillStyle = active ? M.rgba(C.green, 0.08) : C.surface;
    ctx.strokeStyle = active ? C.green : C.border;
    ctx.lineWidth = (active ? 1.6 : 1.0) * s.sc;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, 4 * s.sc); else ctx.rect(x, y, w, h);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = active ? C.green : C.muted;
    ctx.font = M.font(s, 8.0, 600, 'sans');
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('ViT: Attention Head', x + 6 * s.sc, y + 5 * s.sc);

    /* 4 patch tokens on left -> Q*K^T all-to-all arcs -> contextualized tokens on right */
    var xSrc = x + 18 * s.sc, xDst = x + w - 22 * s.sc;
    var yTop = y + 23 * s.sc, stepY = (h - 30 * s.sc) / 3;
    var attnW = [
      [0.5, 0.2, 0.8, 0.1],
      [0.2, 0.7, 0.2, 0.6],
      [0.8, 0.2, 0.6, 0.3],
      [0.1, 0.6, 0.3, 0.7]
    ];
    for (var i = 0; i < 4; i++) {
      for (var j = 0; j < 4; j++) {
        var aw = attnW[i][j];
        if (aw < 0.25) continue;
        ctx.strokeStyle = M.rgba(C.green, aw * 0.72);
        ctx.lineWidth = (0.6 + aw * 1.8) * s.sc;
        ctx.beginPath();
        ctx.moveTo(xSrc + 5 * s.sc, yTop + i * stepY);
        ctx.bezierCurveTo((xSrc + xDst) * 0.5, yTop + i * stepY, (xSrc + xDst) * 0.5, yTop + j * stepY, xDst - 5 * s.sc, yTop + j * stepY);
        ctx.stroke();
      }
    }
    for (var k = 0; k < 4; k++) {
      M.dot(ctx, xSrc, yTop + k * stepY, 3.6 * s.sc, C.surface, C.green, 1.3 * s.sc);
      M.dot(ctx, xDst, yTop + k * stepY, 3.8 * s.sc, C.green, C.surface, 1.0 * s.sc);
    }
  }

  function drawSchematic() {
    var ctx = schemStage.ctx, s = schemStage;
    if (!s.w) return;
    M.clear(ctx, s);
    var padX = 10 * s.sc, fullW = s.w - 2 * padX;

    /* ─── Band 1: Sensory Diet (2 distinct Dog & Cat groups with OR / +, or CLIP pairs) ─── */
    var b1Y = 8 * s.sc, b1H = 78 * s.sc;
    var isInputVary = varyAxis === 'input';
    ctx.fillStyle = isInputVary ? M.rgba(C.orange, 0.08) : C.surface;
    ctx.strokeStyle = isInputVary ? C.orange : C.border;
    ctx.lineWidth = (isInputVary ? 1.6 : 1.0) * s.sc;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(padX, b1Y, fullW, b1H, 4 * s.sc); else ctx.rect(padX, b1Y, fullW, b1H);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = isInputVary ? C.orange : C.muted;
    ctx.font = M.font(s, 8.0, 600, 'sans');
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    var dietTitle = taskMode === 'clip'
      ? '1. SENSORY DIET: Multimodal Vision + Language Pairs (CLIP)'
      : ('1. SENSORY DIET (' + (isInputVary ? 'VARIED' : 'HELD') + '): ' +
         (dietMode === 'dogs' ? 'Dogs only' : dietMode === 'cats' ? 'Cats only' : 'Dogs + Cats'));
    ctx.fillText(dietTitle, padX + 7 * s.sc, b1Y + 5 * s.sc);

    var ty = b1Y + 21 * s.sc, grpH = 50 * s.sc;
    if (taskMode === 'clip') {
      /* Paired image + tokenized caption for Dog and Cat side by side */
      var pairW = (fullW - 20 * s.sc) / 2;
      var tSz = 42 * s.sc, clipY = ty + (grpH - tSz) * 0.5;
      [['dogs', 0, C.blue, ['"a photo', 'of a dog"']], ['cats', 0, C.orange, ['"a photo', 'of a cat"']]].forEach(function (pr, pIdx) {
        var px = padX + 8 * s.sc + pIdx * (pairW + 4 * s.sc);
        drawThumb(ctx, pr[0], pr[1], px, clipY, tSz, pr[2], s);
        ctx.fillStyle = C.green; ctx.font = M.font(s, 9, 600);
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('\u2194', px + tSz + 9 * s.sc, clipY + tSz / 2);

        var chipX = px + tSz + 18 * s.sc, chipW = Math.max(46 * s.sc, pairW - tSz - 22 * s.sc);
        ctx.fillStyle = M.rgba(C.green, 0.12);
        ctx.strokeStyle = M.rgba(C.green, 0.7);
        ctx.lineWidth = 1.0 * s.sc;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(chipX, clipY + 4 * s.sc, chipW, tSz - 8 * s.sc, 4 * s.sc);
        else ctx.rect(chipX, clipY + 4 * s.sc, chipW, tSz - 8 * s.sc);
        ctx.fill(); ctx.stroke();
        ctx.fillStyle = C.ink; ctx.font = M.font(s, 7.6, 600);
        ctx.textAlign = 'center';
        ctx.fillText(pr[3][0], chipX + chipW / 2, clipY + tSz * 0.36);
        ctx.fillText(pr[3][1], chipX + chipW / 2, clipY + tSz * 0.66);
      });
    } else {
      /* Two distinct photo groups: [Dog 1][Dog 2][Dog 3]  OR/+  [Cat 1][Cat 2][Cat 3] */
      var sepW = 34 * s.sc;
      var grpW = (fullW - 14 * s.sc - sepW) * 0.5;
      var leftGrpX = padX + 7 * s.sc;
      var rightGrpX = leftGrpX + grpW + sepW;
      var dogsActive = dietMode !== 'cats';
      var catsActive = dietMode !== 'dogs';
      drawPhotoGroup(ctx, 'dogs', leftGrpX, ty, grpW, grpH, C.blue, dogsActive, s);
      drawPhotoGroup(ctx, 'cats', rightGrpX, ty, grpW, grpH, C.orange, catsActive, s);

      var sepLabel = isInputVary ? 'OR' : '+';
      M.pill(ctx, s, sepLabel, leftGrpX + grpW + sepW * 0.5, ty + grpH * 0.5, {
        align: 'center',
        color: isInputVary ? C.orange : C.muted,
        size: 7.8
      });
    }

    /* ─── Band 2: Computational Architecture Schematics (CNN vs. Transformer) ─── */
    var b2Y = b1Y + b1H + 8 * s.sc, b2H = 90 * s.sc;
    var halfW = (fullW - 8 * s.sc) / 2;
    var cnnActive = varyAxis === 'task' || archMode === 'cnn';
    var vitActive = varyAxis === 'task' || (varyAxis === 'arch' && archMode === 'vit');
    drawConvSchematic(ctx, padX, b2Y, halfW, b2H, cnnActive, s);
    drawAttnSchematic(ctx, padX + halfW + 8 * s.sc, b2Y, halfW, b2H, vitActive, s);

    /* ─── Band 3: Controlled Open-Source Model Zoo Comparison (2-line entries) ─── */
    var b3Y = b2Y + b2H + 8 * s.sc, b3H = s.h - b3Y - 8 * s.sc;
    ctx.fillStyle = C.surface;
    ctx.strokeStyle = C.border;
    ctx.lineWidth = 1.1 * s.sc;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(padX, b3Y, fullW, b3H, 4 * s.sc); else ctx.rect(padX, b3Y, fullW, b3H);
    ctx.fill(); ctx.stroke();

    var zooMap = {
      task: {
        head: '3. VARY OBJECTIVE (Hold Diet + Architecture):',
        arms: [
          ['Supervised Learning (e.g. Torchvision ResNet18)', 'Explicit category labels pull Dog & Cat poles apart', C.blue],
          ['Self-Supervised Contrastive Learning (e.g. SimCLR)', 'Instance-augmentation invariance without category labels', C.green],
          ['Vision-Language Contrastive Learning (e.g. CLIP)', 'Joint image\u2013caption cosine alignment on the unit sphere', C.orange]
        ]
      },
      arch: {
        head: '3. VARY ARCHITECTURE (Hold Dogs+Cats Diet + Supervised):',
        arms: [
          ['ResNet18 / ConvNeXt (e.g. timm)', 'Local 2D sliding filters build translation-equivariant maps', C.blue],
          ['Vision Transformer ViT-B/16 (e.g. timm)', 'Global multi-head attention routes context across all patches', C.green]
        ]
      },
      input: {
        head: '3. VARY SENSORY DIET (Hold CNN + Contrastive Objective):',
        arms: [
          ['Open-IPCL \u00b7 Narrow / Single-Domain Diet (Dogs OR Cats)', 'Specializes fine-grained within-class distinctions', C.blue],
          ['Open-IPCL \u00b7 Diverse Multi-Category Diet (Dogs + Cats)', 'Carves broader between-category axes (Konkle & Alvarez, 2022)', C.orange]
        ]
      }
    }[varyAxis];

    ctx.fillStyle = C.ink; ctx.font = M.font(s, 8.1, 600, 'sans');
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(zooMap.head, padX + 7 * s.sc, b3Y + 6 * s.sc);

    var nArms = zooMap.arms.length;
    var rowStep = Math.max(25 * s.sc, (b3H - 23 * s.sc) / nArms);
    for (var ai = 0; ai < nArms; ai++) {
      var ay = b3Y + 21 * s.sc + ai * rowStep;
      var arm = zooMap.arms[ai];
      M.dot(ctx, padX + 11 * s.sc, ay + 4.5 * s.sc, 3.0 * s.sc, arm[2]);
      ctx.fillStyle = C.ink; ctx.font = M.font(s, 7.9, 600, 'sans');
      ctx.fillText(arm[0], padX + 19 * s.sc, ay);
      ctx.fillStyle = C.muted; ctx.font = M.font(s, 7.4, 400, 'sans');
      ctx.fillText(arm[1], padX + 19 * s.sc, ay + 11.2 * s.sc);
    }
  }

  function drawAll() {
    if (!ready) return;
    drawSchematic(); drawSpherePanel();
    var holdSummary = varyAxis === 'task'
      ? 'Sensory Diet (Dogs+Cats) + Architecture (CNN/ViT)'
      : varyAxis === 'arch'
        ? 'Sensory Diet (Dogs+Cats) + Objective (Supervised)'
        : 'Architecture (CNN) + Objective (Contrastive)';
    read.innerHTML = '<span class="k">varied pressure</span> ' + (varyAxis === 'task' ? 'Task Objective (' + taskMode + ')' : varyAxis === 'arch' ? 'Architecture (' + archMode.toUpperCase() + ')' : 'Sensory Diet (' + dietMode + ')') +
      '<span class="sep">&middot;</span><span class="k">held constant</span> ' + holdSummary +
      '<div class="wread-interp"><span class="g">Holding two pressures constant while varying the third turns open-source model zoos into controlled experiments.</span></div>';
  }

  schemStage.draw = drawAll; sphereStage.draw = drawAll;
  ready = true;
  tr.show(60);
  schemStage.request(); sphereStage.request();
  return { destroy: function () { tr.pause(); } };
});
